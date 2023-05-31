import argparse
import os
import warnings
import glob
import json
import mmcv
import torch
import sys
sys.path.append('/home/jovyan/berkiu/instancesegmentation/mmdetection/')
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import init_detector, inference_detector

from mmdet.apis import multi_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
# import aicrowd_helpers
import os.path as osp
import traceback
import pickle
import shutil
import tempfile
import time
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmdet.core import encode_mask_results

import uuid

# TEST_IMAGES_PATH = "/mnt/public/xxx/imrec/data/val/images"

def create_test_predictions(images_path):
    test_predictions_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".json")
	
    annotations = {'categories': [], 'info': {}, 'images': []}
    for item in glob.glob(images_path+'/*.jpg'):
        image_dict = dict()
        img = mmcv.imread(item)
        height,width,__ = img.shape
        id = int(os.path.basename(item).split('.')[0])
        image_dict['id'] = id
        image_dict['file_name'] = os.path.basename(item)
        image_dict['width'] = width
        image_dict['height'] = height
        annotations['images'].append(image_dict)
    annotations['categories'] = json.loads(open("classes.json").read())
    json.dump(annotations, open(test_predictions_file.name, 'w'))

    return test_predictions_file

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        # aicrowd_helpers.execution_progress({"image_ids" : [i]})
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # Perform RLE encode for masks
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--data', help='test data folder path')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--out_file', help='output result file')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--type', type=str, choices=['val', 'test'], default='test')
    parser.add_argument('--reduce_ms', action='store_true',
        help='Whether to reduce the multi-scale aug')
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def reduce_multiscale_TTA(cfg):
    '''
    Keep only 1st and last image sizes from Multi-Scale TTA
    
    @input
    cfg -> Configuration file
    '''

    scale = cfg.data.test.pipeline[1]['img_scale']
    if len(scale) > 2:
        new_scale = [scale[0], scale[-1]]
        cfg.data.test.pipeline[1]['img_scale'] = new_scale   
    return cfg

def main():
    ########################################################################
    # Register Prediction Start
    ########################################################################

    # aicrowd_helpers.execution_start()
    args = parse_args()
    data_folder = args.data
    # Create annotations if not already created
    test_predictions_file = create_test_predictions(data_folder)
    
    # Load annotations
    with open(test_predictions_file.name) as f:
        annotations = json.loads(f.read())

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    JSONFILE_PREFIX="predictions_{}".format(str(uuid.uuid4())) 
    # import modules present in list of strings.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 2
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = test_predictions_file.name
    cfg.data.test.img_prefix = data_folder

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    cfg.data.test.ann_file = test_predictions_file.name
    cfg.data.test.img_prefix = data_folder
        
    # if args.reduce_ms:
    #     print("Reduce multi-scale TTA")
    #     cfg = reduce_multiscale_tta(cfg)
    #     print(cfg.data.test.pipeline[1]['img_scale'])
        
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    
    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    print(dataset)
    dataset.cat_ids = [category["id"] for category in annotations["categories"]]
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    # model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.model.test_cfg)
    model = init_detector(args.config,args.checkpoint,device='cuda:0')

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # checkpoint = load_checkpoint(model, args.checkpoint, map_location='cuda')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    model.CLASSES = [category['name'] for category in annotations['categories']]
    # if 'CLASSES' in checkpoint['meta']:
        # model.CLASSES = checkpoint['meta']['CLASSES']
    # else:
        # model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))
    
    # consolidate_results(["predictions.segm.json"], 'test_predictions.json', args.out_file)
    ########################################################################
    # Register Prediction Complete
    ########################################################################
    # aicrowd_helpers.execution_success({
    #     "predictions_output_path" : args.out_file
    # })
    print("\nAICrowd register complete")
    # preds = []
    # with open("predictions.segm.json", "r") as pred_file:
    #     preds.extend(json.loads(pred_file.read()))
    # print(preds)
    JSONFILE_PREFIX = args.eval_options['jsonfile_prefix']
    shutil.move("{}.segm.json".format(JSONFILE_PREFIX), args.out_file)
    os.remove("{}.bbox.json".format(JSONFILE_PREFIX))
        
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        error = traceback.format_exc()
        print(error)