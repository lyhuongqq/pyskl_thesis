# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import torch
import torch.distributed as dist
from mmcv import Config
from mmcv.runner import get_dist_info
import json

from pyskl.datasets import build_dataset
from pyskl.models import build_model
from pyskl.utils import get_root_logger, collect_env
from pyskl.apis import single_gpu_test

def parse_args():
    parser = argparse.ArgumentParser(description='Test a recognizer')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='path to the model checkpoint file')
    parser.add_argument('--out', help='path to save the test results (in JSON format)')
    parser.add_argument('--eval', nargs='+', help='evaluation metrics like top_k_accuracy, mean_class_accuracy')
    parser.add_argument('--launcher', choices=['pytorch', 'slurm'], default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    return args

def main():
    args = parse_args()

    # Load configuration file
    cfg = Config.fromfile(args.config)

    # Set up GPU device
    rank, world_size = get_dist_info()
    cfg.gpu_ids = [1]  # Use GPU 1; modify as needed

    # Initialize the logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'test_{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.get('log_level', 'INFO'))

    # Log environment info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

    # Build the model
    model = build_model(cfg.model)
    model.eval()

    # Load checkpoint
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info(f'Loaded checkpoint from {args.checkpoint}')

    # Build dataset for evaluation
    dataset = build_dataset(cfg.data.test)

    # Conduct evaluation
    if args.eval:
        results = single_gpu_test(model, dataset, show=False)
        eval_results = dataset.evaluate(results, metrics=args.eval)

        # Log evaluation results
        logger.info(f'Evaluation results: {eval_results}')
        
        # Save results to JSON if required
        if args.out:
            with open(args.out, 'w') as f:
                json.dump(eval_results, f)
            logger.info(f'Results saved to {args.out}')
    else:
        logger.warning('No evaluation metrics specified, skipping evaluation.')

if __name__ == '__main__':
    main()
