import os.path as osp

import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pyskl.datasets.base import BaseDataset
from pyskl.datasets.builder import DATASETS

# Check if the VideoDataset is already registered
if 'VideoDataset' not in DATASETS.module_dict:
    @DATASETS.register_module()
    class VideoDataset(BaseDataset):
        """Video dataset for action recognition.

        The dataset loads raw videos and apply specified transforms to return a
        dict containing the frame tensors and other information.

        The ann_file is a text file with multiple lines, and each line indicates
        a sample video with the filepath and label, which are split with a
        whitespace.

        Args:
            ann_file (str): Path to the annotation file.
            pipeline (list[dict | callable]): A sequence of data transforms.
            start_index (int): Specify a start index for frames in consideration of
                different filename format. However, when taking videos as input,
                it should be set to 0, since frames loaded from videos count
                from 0. Default: 0.
            **kwargs: Keyword arguments for ``BaseDataset``.
        """

        def __init__(self, ann_file, pipeline, start_index=0, **kwargs):
            super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)

        def load_annotations(self):
            """Load annotation file to get video information."""
            if self.ann_file.endswith('.json'):
                return self.load_json_annotations()

            video_infos = []
            with open(self.ann_file, 'r') as fin:
                for line in fin:
                    line_split = line.strip().split()
                    if self.multi_class:
                        assert self.num_classes is not None
                        filename, label = line_split[0], line_split[1:]
                        label = list(map(int, label))
                    else:
                        filename, label = line_split
                        label = int(label)
                    filename = osp.join(self.data_prefix, filename)
                    video_infos.append(dict(filename=filename, label=label))
            return video_infos

def main():
    ann_file = r'D:\pyskl-main\pyskl-main\pyskl\datasets\action_test.txt'  # Replace with your annotation file path
    pipeline = [        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=1),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='RandomResizedCrop'),
        dict(type='Resize', scale=(224, 224), keep_ratio=False),
        dict(type='Flip', flip_ratio=0.5),
        dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label']),
    ]  # Replace with your pipeline
    dataset = VideoDataset(ann_file, pipeline)
    annotations = dataset.load_annotations()
    print(annotations)

if __name__ == '__main__':
    main()
