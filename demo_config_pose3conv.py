model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        in_channels=21,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride=(1, 1),
        pool1_stride=(1, 1),
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2)),
    cls_head=dict(type='I3DHead', in_channels=512, num_classes=6, dropout=0.5),
    test_cfg=dict(average_clips='prob'))
dataset_type = 'PoseDataset'
#ann_file = '/root/pyskl_thesis/hand_pose_dataset_aug_6.pkl'
#hand_kp = [
#    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
#]


test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=10, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(
        type='GeneratePoseTarget', with_kp=True, with_limb=False,
        double=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]