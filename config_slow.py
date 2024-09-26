model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        in_channels=21,  # Updated to 21 for single hand keypoints
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride=(1, 1),
        pool1_stride=(1, 1),
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2)),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=6,  # Number of classes remains 6
        dropout=0.5),
    test_cfg=dict(average_clips='prob')
)

dataset_type = 'PoseDataset'
ann_file = r"/root/pyskl_thesis/hand_pose_dataset_0.pkl"  # Path to your hand pose dataset pickle file

#left_kp = list(range(21))  # Left hand keypoints [0, 1, 2, ..., 20]
#right_kp = list(range(21, 42))  # Right hand keypoints [21, 22, 23, ..., 41]

hand_kp = list(range(21)) # 21 keypoints for a single hand

train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=10),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5), #left_kp=left_kp, right_kp=right_kp), # No left/right keypoint flipping necessary for single hand
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False, double=False), #, left_kp=left_kp, right_kp=right_kp), # No need for double=False with single hand
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=16,  # Adjust based on your GPU memory
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(type=dataset_type, ann_file=ann_file, split='train', pipeline=train_pipeline)),
    val=dict(type=dataset_type, ann_file=ann_file, split='val', pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file, split='val', pipeline=test_pipeline)
)

# optimizer
optimizer = dict(type='SGD', lr=0.4, momentum=0.9, weight_decay=0.0003)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 24
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/posec3d/test_slow_mp'

# device setting for GPU
device = 'cuda'
gpu_ids = range(1)  # Use one GPU
