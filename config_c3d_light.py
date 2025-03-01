model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='C3D',
        in_channels=21,
        base_channels=32,
        num_stages=3,
        temporal_downsample=False),
    cls_head=dict(
        type='I3DHead',
        in_channels=256,
        num_classes=6,
        dropout=0.5),
    test_cfg=dict(average_clips='prob'))

dataset_type = 'PoseDataset'
#ann_file = r"/root/pyskl_thesis/hand_pose_dataset_val.pkl"  # Path to your hand pose dataset pickle file
#ann_file = r"/root/pyskl_thesis/hand_pose_dataset_aug_6.pkl""
ann_file = r"/root/pyskl_thesis/smoterandom_csv_paper_train_split_frame_wise.pkl"
ann_file1 = r"/root/pyskl_thesis/Jasmoterandom_csv_paper_train_split_frame_wise.pkl"

train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=1,frame_interval=1,num_clips=1),
    #dict(type='UniformSampleFramesEach', clip_len=5,frame_interval=1,num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5), #, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=20, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    #dict(type='UniformSampleFrames', clip_len=20, num_clips=10),
    dict(type='UniformSampleFrames', clip_len=1, frame_interval=1, num_clips=2),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False) ,# double=True, left_kp=left_kp, right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(type=dataset_type, ann_file=ann_file, split='train', pipeline=train_pipeline)),
    val=dict(type=dataset_type, ann_file=ann_file, split='val', pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file1, split='val', pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.05, momentum=0.05, weight_decay=0.0003)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 50
checkpoint_config = dict(interval=1)
#evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy', 'mean_average_precision','binary_precision_recall_curve',
                                        'precision', 'recall', 'f1_score','accuracy','calculate_accuracy','evaluate_class_metrics'
                                        'confusion_matrix','calculate_weighted_metrics'], topk=(1, 5))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/c3d_paper_smote_random'
