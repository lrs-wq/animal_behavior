"""Fine-tune RTMPose-m with real local-camera detector proposals.

Leakage boundary:
* camera train/val/test are grouped by source clip by the dataset builder;
* all legacy non-v4 clean crops are train-only;
* load ONLY the official AP-10K base checkpoint below.  The previous
  clean516 best checkpoint has seen every v4 label, including the new
  validation/test clips, and is forbidden for uncontaminated metrics.
"""

_base_ = [
    '/opt/miniconda3/envs/rtmpose/lib/python3.10/site-packages/mmpose/.mim/'
    'configs/animal_2d_keypoint/rtmpose/ap10k/'
    'rtmpose-m_8xb64-210e_ap10k-256x256.py'
]

data_root = (
    '/data/Eating_behavior/rtmpose/'
    'cat_pose_local_camera_fullframe_proposals_v1/'
)
work_dir = (
    '/data/Eating_behavior/rtmpose/runs/'
    'cat_pose_local_camera_proposals_rtmpose_m_ap10k_v1/'
)

dataset_type = 'AP10KDataset'
data_mode = 'topdown'
max_epochs = 150
stage2_num_epochs = 20
base_lr = 2.5e-4
train_batch_size = 32

train_cfg = dict(max_epochs=max_epochs, val_interval=5)
optim_wrapper = dict(
    optimizer=dict(lr=base_lr, weight_decay=0.02),
    clip_grad=dict(max_norm=10.0, norm_type=2))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=100),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=10,
        end=max_epochs,
        T_max=max_epochs - 10,
        by_epoch=True,
        convert_to_iter_based=True),
]
auto_scale_lr = dict(enable=False, base_batch_size=512)
# Launch training with CUBLAS_WORKSPACE_CONFIG=:4096:8 so CUDA linear layers
# can honor this deterministic setting instead of failing on the first step.
randomness = dict(seed=21, deterministic=True)
env_cfg = dict(cudnn_benchmark=False)

codec = dict(
    type='SimCCLabel',
    input_size=(256, 256),
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)
backend_args = dict(backend='local')

# The proposal builder gates human points against this same 1.25 expansion.
# No half-body transform is used because the local labels are sparse and paws
# or tail points must not be silently cropped away.
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale', padding=1.25),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.03,
        scale_factor=[0.90, 1.15],
        rotate_factor=15),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.05),
            dict(type='MedianBlur', p=0.05),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

# Final 20 epochs switch to a low-noise pipeline for precise proposal-aligned
# localization.  The PipelineSwitchHook below is intentionally present in this
# child config; replacing custom_hooks without it was the previous bug.
train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale', padding=1.25),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.01,
        scale_factor=[0.98, 1.08],
        rotate_factor=5),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale', padding=1.25),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs'),
]

# MultiSourceSampler fixes every batch at 16 generic + 16 local-camera samples.
# This deliberately oversamples the much smaller camera-train subset.  The
# builder records the expected repeat factor in build_report.json.
train_datasets = [
    dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/generic_train.json',
        data_prefix=dict(img='images/generic_train/'),
        pipeline=[]),
    dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        pipeline=[]),
]

train_dataloader = dict(
    _delete_=True,
    batch_size=train_batch_size,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(
        type='MultiSourceSampler',
        batch_size=train_batch_size,
        source_ratio=[1, 1],
        shuffle=True,
        round_up=True,
        seed=21),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(
            from_file=(
                '/opt/miniconda3/envs/rtmpose/lib/python3.10/site-packages/'
                'mmpose/.mim/configs/_base_/datasets/ap10k.py')),
        datasets=train_datasets,
        pipeline=train_pipeline,
        test_mode=False))

val_dataloader = dict(
    _delete_=True,
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/val.json',
        bbox_file=data_root + 'annotations/val_detector_bboxes.json',
        data_prefix=dict(img='images/val/'),
        test_mode=True,
        pipeline=val_pipeline))

test_dataloader = dict(
    _delete_=True,
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/test.json',
        bbox_file=data_root + 'annotations/test_detector_bboxes.json',
        data_prefix=dict(img='images/test/'),
        test_mode=True,
        pipeline=val_pipeline))

# Val/test inference reads the frozen production proposals from bbox_file.  The
# COCO annotation bbox/area stays tied to the source crop annotation instead of
# being overwritten by the evaluated detector.  Those source boxes have mixed
# manual/prediction provenance, so coco/AP is for internal model selection only
# and must not be presented as a standard AP-10K benchmark result.
val_evaluator = dict(
    type='CocoMetric', ann_file=data_root + 'annotations/val.json')
test_evaluator = dict(
    type='CocoMetric', ann_file=data_root + 'annotations/test.json')

# Official RTMPose-m AP-10K base only.
# Expected SHA256:
# 896e3665d849ef7eb9b6ec0995955796cc9810f024fa0aa0bdc18acb0d68bf52
load_from = '/root/rtmpose/checkpoints/rtmpose-m_ap10k_256x256.pth'
resume = False

default_hooks = dict(
    logger=dict(interval=10),
    checkpoint=dict(
        interval=5,
        save_best='coco/AP',
        rule='greater',
        max_keep_ckpts=3,
        save_last=True))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2),
]

# Do not add EarlyStoppingHook before epoch 130: it could terminate training
# before PipelineSwitchHook ever activates.  Checkpoint selection still uses
# the internal validation coco/AP after the full scheduled training run.
