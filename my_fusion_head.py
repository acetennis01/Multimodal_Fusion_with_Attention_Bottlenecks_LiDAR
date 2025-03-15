# my_fusion_head.py

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models.builder import HEADS, build_loss

@HEADS.register_module()
class FusionClassifierHead(BaseModule):
    """
    Minimal classification head: takes 'fused_feat' [B, dim] and predicts classes.
    """
    def __init__(self,
                 in_channels=768,
                 num_classes=3,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.num_classes = num_classes
        self.fc = nn.Linear(in_channels, num_classes)
        self.loss_cls = build_loss(loss_cls)

    def forward(self, x, labels=None):
        """
        Args:
            x (Tensor|dict): either a dict with {'fused_feat': [B, dim]}
                             or directly a tensor [B, dim]
            labels (Tensor|None): if given, compute and return loss; else return logits
        """
        if isinstance(x, dict):
            x = x['fused_feat']  # => [B, dim]

        logits = self.fc(x)  # => [B, num_classes]

        if labels is not None:
            loss_cls = self.loss_cls(logits, labels)
            return dict(loss_cls=loss_cls)
        else:
            return logits
