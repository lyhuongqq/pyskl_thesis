# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa: E722
import argparse
import mmcv
import os
import os.path as osp
import time
import torch
from mmcv import Config
from mmcv import digit_version as dv
from mmcv import load
from mmcv.runner import get_dist_info, load_checkpoint
from mmcv.fileio.io import file_handlers

from pyskl.datasets import build_dataloader, build_dataset
from pyskl.models import build_model
from pyskl.utils import cache_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='pyskl test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('-C', '--checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--out', default=None, help='output result file in pkl/yaml/json format')
    parser.add_argument('--fuse-conv-bn', action='store_true',
                        help='Whether to fuse conv and bn, this will slightly increase inference speed')
    parser.add_argument('--eval', type=str, nargs='+', default=['top_k_accuracy', 'mean_class_accuracy'],
                        help='evaluation metrics, which depends on the dataset, e.g., "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument('--tmpdir', help='tmp directory used for collecting results from multiple workers')
    parser.add_argument('--average-clips', choices=['score', 'prob', None], default=None,
                        help='average type when averaging test clips')
    parser.add_argument('--launcher', choices=['pytorch', 'slurm'], default='pytorch', help='job launcher')
    parser.add_argument('--compile', action='store_true',
                        help='whether to compile the model before training / testing (only available in pytorch 2.0)')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    return args


def inference_pytorch(args, cfg, data_loader):
    """Get predictions using PyTorch models running on CPU."""
    if args.average_clips is not None:
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg', dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # build the model and load checkpoint
    model = build_model(cfg.model)
    
    if dv(torch.__version__) >= dv('2.0.0') and args.compile:
        model = torch.compile(model)

    # Load checkpoint
    if args.checkpoint is None:
        work_dir = cfg.work_dir
        args.checkpoint = osp.join(work_dir, 'latest.pth')
        assert osp.exists(args.checkpoint), "Checkpoint file not found."

    args.checkpoint = cache_checkpoint(args.checkpoint)
    load_checkpoint(model, args.checkpoint, map_location='cpu')  # Force loading on CPU

    # No need for MMDistributedDataParallel on CPU, running inference directly
    model = model.to('cpu')  # Ensure the model is on CPU

    outputs = []  # Placeholder for outputs
    model.eval()

    # Process batches
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        outputs.append(result)

    return outputs


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # Define output file
    out = osp.join(cfg.work_dir, 'result.pkl') if args.out is None else args.out

    # Load eval_config from cfg
    eval_cfg = cfg.get('evaluation', {})
    keys = ['interval', 'tmpdir', 'start', 'save_best', 'rule', 'by_epoch', 'broadcast_bn_buffers']
    for key in keys:
        eval_cfg.pop(key, None)
    if args.eval:
        eval_cfg['metrics'] = args.eval

    mmcv.mkdir_or_exist(osp.dirname(out))
    _, suffix = osp.splitext(out)
    #assert suffix[1:] in mmcv.fileio.file_handlers, \
    #    ('The format of the output file should be json, pickle or yaml')
   
    assert suffix[1:] in file_handlers, ('The format of the output file should be json, pickle or yaml')
    # Ensure the model runs on CPU
    torch.backends.cudnn.benchmark = False  # Not needed for CPU
    cfg.data.test.test_mode = True

    # Build the dataset and dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),  # Adjust for CPU
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),  # Adjust for CPU
        shuffle=False
    )
    dataloader_setting = dict(dataloader_setting, **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # Perform inference
    outputs = inference_pytorch(args, cfg, data_loader)

    # Save the outputs
    if len(outputs) > 0:
        print(f'\nwriting results to {out}')
        dataset.dump_results(outputs, out=out)
        if eval_cfg:
            eval_res = dataset.evaluate(outputs, **eval_cfg)
            for name, val in eval_res.items():
                print(f'{name}: {val:.04f}') #val
    else:
        print('No valid outputs were generated.')


if __name__ == '__main__':
    main()
