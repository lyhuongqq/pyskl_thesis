# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa: E722
import copy
import mmcv
import numpy as np
import os
import os.path as osp
import torch
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from mmcv.utils import print_log
from torch.utils.data import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support

import sys
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pyskl.smp import auto_mix2
#from ..core import mean_average_precision, mean_class_accuracy, top_k_accuracy
from pyskl.core import mean_average_precision, mean_class_accuracy, top_k_accuracy
#from .pipelines import Compose
from pyskl.datasets.pipelines import Compose


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for datasets.

    All datasets to process video should subclass it.
    All subclasses should overwrite:

    - Methods:`load_annotations`, supporting to load information from an
    annotation file.
    - Methods:`prepare_train_frames`, providing train data.
    - Methods:`prepare_test_frames`, providing test data.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str): Path to a directory where videos are held. Default: ''.
        test_mode (bool): Store True when building test or validation dataset. Default: False.
        multi_class (bool): Determines whether the dataset is a multi-class dataset. Default: False.
        num_classes (int | None): Number of classes of the dataset, used in multi-class datasets. Default: None.
        start_index (int): Specify a start index for frames in consideration of different filename format. However,
            if taking videos as input, it should be set to 0, since frames loaded from videos count from 0. Default: 1.
        modality (str): Modality of data. Support 'RGB', 'Flow', 'Audio'. Default: 'RGB'.
        memcached (bool): Whether keypoint is cached in memcached. If set as True, will use 'frame_dir' as the key to
            fetch 'keypoint' from memcached. Default: False.
        mc_cfg (tuple): The config for memcached client, only applicable if `memcached==True`.
            Default: ('localhost', 22077).
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix='',
                 test_mode=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 memcached=False,
                 mc_cfg=('localhost', 22077)):
        super().__init__()

        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        # Note: Currently, memcached only works for PoseDataset
        self.memcached = memcached
        self.mc_cfg = mc_cfg
        self.cli = None

        self.pipeline = Compose(pipeline)
        self.video_infos = self.load_annotations()

    @abstractmethod
    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""

    # json annotations already looks like video_infos, so for each dataset,
    # this func should be the same
    def load_json_annotations(self):
        """Load json annotation file to get video information."""
        video_infos = mmcv.load(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            path_value = osp.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value
            if self.multi_class:
                assert self.num_classes is not None
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = video_infos[i]['label'][0]
        return video_infos

    def parse_by_class(self):
        video_infos_by_class = defaultdict(list)
        for item in self.video_infos:
            label = item['label']
            video_infos_by_class[label].append(item)
        return video_infos_by_class

    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 metric_options=dict(top_k_accuracy=dict(topk=(1, 5))),
                 logger=None,
                 **deprecated_kwargs):
        """Perform evaluation for common datasets, including precision, recall, F1-score.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed. Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options. Default: ``dict(top_k_accuracy=dict(topk=(1, 5)))``.
            logger (logging.Logger | None): Logger for recording. Default: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.

        Returns:
            dict: Evaluation results dict, including precision, recall, and f1-score.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['top_k_accuracy', 'mean_class_accuracy', 'mean_average_precision', 'precision', 'recall', 'f1_score']

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        gt_labels = [ann['label'] for ann in self.video_infos]

        # Ensure metric_options is deepcopied to avoid mutable default
        metric_options = copy.deepcopy(metric_options)

        # Evaluate each metric
        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy', {}).setdefault('topk', (1, 5))
                if isinstance(topk, int):
                    topk = (topk, )
                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc'] = acc
                    print_log(f'\ntop{k}_acc\t{acc:.4f}', logger=logger)

            if metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                print_log(f'\nmean_acc\t{mean_acc:.4f}', logger=logger)

            if metric == 'mean_average_precision':
                gt_labels_arrays = [self.label2array(self.num_classes, label) for label in gt_labels]
                mAP = mean_average_precision(results, gt_labels_arrays)
                eval_results['mean_average_precision'] = mAP
                print_log(f'\nmean_average_precision\t{mAP:.4f}', logger=logger)

            if metric in ['precision', 'recall', 'f1_score']:
                pred_labels = np.argmax(results, axis=1)
                precision, recall, f1, _ = precision_recall_fscore_support(gt_labels, pred_labels, average=None, labels=np.arange(self.num_classes))

                for i in range(self.num_classes):
                    eval_results[f'{metric}_class_{i}'] = eval(metric[i])
                    print_log(f'\n{metric}_class_{i}\t{eval_results[f"{metric}_class_{i}"]:.4f}', logger=logger)

                # Compute overall metrics across all classes
                precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(gt_labels, pred_labels, average='macro')
                eval_results['precision_macro'] = precision_macro
                eval_results['recall_macro'] = recall_macro
                eval_results['f1_macro'] = f1_macro
                print_log(f'\nprecision_macro\t{precision_macro:.4f}', logger=logger)
                print_log(f'\nrecall_macro\t{recall_macro:.4f}', logger=logger)
                print_log(f'\nf1_macro\t{f1_macro:.4f}', logger=logger)

        return eval_results

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return mmcv.dump(results, out)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        if self.memcached and 'key' in results:
            from pymemcache import serde
            from pymemcache.client.base import Client

            if self.cli is None:
                self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)
            key = results.pop('key')
            try:
                pack = self.cli.get(key)
            except:
                self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)
                pack = self.cli.get(key)
            if not isinstance(pack, dict):
                raw_file = results['raw_file']
                data = mmcv.load(raw_file)
                pack = data[key]
                for k in data:
                    try:
                        self.cli.set(k, data[k])
                    except:
                        self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)
                        self.cli.set(k, data[k])
            for k in pack:
                results[k] = pack[k]

        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        results['test_mode'] = self.test_mode
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        if self.memcached and 'key' in results:
            from pymemcache import serde
            from pymemcache.client.base import Client

            if self.cli is None:
                self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)
            key = results.pop('key')
            try:
                pack = self.cli.get(key)
            except:
                self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)
                pack = self.cli.get(key)
            if not isinstance(pack, dict):
                raw_file = results['raw_file']
                data = mmcv.load(raw_file)
                pack = data[key]
                for k in data:
                    try:
                        self.cli.set(k, data[k])
                    except:
                        self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)
                        self.cli.set(k, data[k])
            for k in pack:
                results[k] = pack[k]

        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        results['test_mode'] = self.test_mode
        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        return self.prepare_test_frames(idx) if self.test_mode else self.prepare_train_frames(idx)
