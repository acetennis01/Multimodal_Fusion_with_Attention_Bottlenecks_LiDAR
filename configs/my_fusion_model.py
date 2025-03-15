# configs/my_fusion_model.py

# 1) Dataset settings
point_cloud_range = [-50, -50, -5, 50, 50, 3]  # example
train_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=4,
         use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # You can add data augmentation steps here (ObjectSample, RandomFlip3D, etc.)
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.78539816, 0.78539816],
         scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),

    # Our custom pipeline step to make a pseudo‐image
    dict(type='PointsToPseudoImage',
         image_size=(256, 256),
         point_cloud_range=point_cloud_range),

    # Finally pack the inputs
    dict(type='Pack3DDetInputs',
         keys=['points', 'lidar_pseudo_img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type='KittiDataset',  # or your dataset type
        data_root='data/kitti/',
        ann_file='kitti_infos_train.pkl',
        pipeline=train_pipeline,
        # other dataset params...
    )
)

# 2) Model settings
model = dict(
    type='Base3DDetector',  # or any custom 3D model base that calls backbone+head
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        # Possibly mean/std normalization for 'img' if you have images
    ),

    backbone=dict(
        type='FusionBackbone',
        num_latents=4,
        dim=768,
        num_blocks=4,
        pretrained_vit=False  # set True if you have camera data
    ),
    neck=None,  # skip if doing pure classification

    bbox_head=dict(
        type='FusionClassifierHead',
        in_channels=768,
        num_classes=3,
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0)
    )
)

# 3) Training settings (optimizer, runner, etc.)
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 1000,
        by_epoch=False,
        begin=0,
        end=500)
]

max_epochs = 12
train_cfg = dict()
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    checkpoint=dict(interval=1),
    logger=dict(interval=10)
)

load_from = None  # or specify a checkpoint
resume = False
