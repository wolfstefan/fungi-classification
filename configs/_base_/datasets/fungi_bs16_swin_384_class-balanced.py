_base_ = ['./pipelines/rand_aug.py']

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
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(438, -1),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=384),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-2,
        dataset=dict(
            type=dataset_type,
            data_prefix='data/fungiclef2022/DF20',
            ann_file='data/fungiclef2022/DF20-train_metadata.csv',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_prefix='data/fungiclef2022/DF20',
        ann_file='data/fungiclef2022/DF20-val_metadata.csv',
        pipeline=test_pipeline),
    test=dict(
        type='FungiTest',
        data_prefix='data/fungiclef2022/DF21',
        ann_file='data/fungiclef2022/FungiCLEF2022_test_metadata.csv',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['f1_score', 'accuracy'], save_best='f1_score', rule='greater')
