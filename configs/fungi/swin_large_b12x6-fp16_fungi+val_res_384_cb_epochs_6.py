_base_ = [
    '../_base_/models/swin_transformer/large_384_aug.py', '../_base_/datasets/fungi_bs16_swin_384_class-balanced.py',
    '../_base_/schedules/fungi_bs64_adamw_swin.py', '../_base_/default_runtime.py'
]

# model settings
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin-large_3rdparty_in21k-384px.pth'  # noqa
model = dict(
    type='ImageClassifier',
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone')),
    head=dict(num_classes=1604, ),
    train_cfg=dict(_delete_=True))

# dataset settings
dataset_type = 'Fungi'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=384,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies={{_base_.rand_increasing_policies}},
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in img_norm_cfg['mean'][::-1]],
            interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
data = dict(
    samples_per_gpu=12,
    train=dict(
        dataset=[
            dict(
                type=dataset_type,
                data_prefix='data/fungiclef2022/DF20',
                ann_file='data/fungiclef2022/DF20-train_metadata.csv',
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                data_prefix='data/fungiclef2022/DF20',
                ann_file='data/fungiclef2022/DF20-val_metadata.csv',
                pipeline=train_pipeline),
        ]))

evaluation = dict(interval=1)
checkpoint_config = dict(interval=1, max_keep_ckpts=2)

optimizer = dict(lr=5e-4 * 72 / 512)

# learning policy
lr_config = dict(warmup_iters=4200, warmup_by_epoch=False)
runner = dict(max_epochs=6)

log_config = dict(interval=20)  # log every 20 intervals

fp16 = dict(loss_scale='dynamic')