# my_fusion_backbone.py

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models import BACKBONES  # or from mmdet3d.models import BACKBONES
import timm

# Import your modules
from models.pet_modules import AdaptFormer
from models.visual_model import PointCloudEncoder

@BACKBONES.register_module()
class FusionBackbone(BaseModule):
    """
    A custom backbone that:
      - Takes 'lidar_pseudo_img' from the pipeline
      - (Optionally) takes 'img' if you have camera data
      - Fuses them via AdaptFormer
      - Returns a single fused feature vector [B, dim] in a dict
    """
    def __init__(self,
                 num_latents=4,
                 dim=768,
                 num_blocks=4,
                 pretrained_vit=True,
                 init_cfg=None):
        super().__init__(init_cfg)

        # 1) Build the LiDAR pseudo‐image encoder
        self.pc_encoder = PointCloudEncoder(dim=dim)

        # 2) Build a Vision Transformer for camera input (if needed)
        self.use_camera = pretrained_vit  # or some other condition
        if self.use_camera:
            self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
            self.vit.pre_logits = nn.Identity()
            self.vit.head = nn.Identity()
        else:
            self.vit = None

        # 3) Build AdaptFormer blocks for cross‐modal fusion
        blocks = []
        for _ in range(num_blocks):
            blocks.append(AdaptFormer(num_latents=num_latents, dim=dim))
        self.fusion_blocks = nn.Sequential(*blocks)

        # 4) Final LN (reuse ViT norm) or a new LayerNorm
        if self.use_camera:
            self.post_norm = self.vit.norm  # typical LN
        else:
            self.post_norm = nn.LayerNorm(dim)

    def forward(self, inputs):
        """
        Args:
            inputs (dict): Typically the packed data from the pipeline.
                           Should contain 'lidar_pseudo_img' of shape [B, 1, H, W] or [B, C, H, W].
                           If using camera, also 'img' of shape [B, T, 3, H, W] or [B, 3, H, W].
        Returns:
            dict: e.g. {'fused_feat': [B, dim]}
        """

        # 1) Get LiDAR pseudo‐image
        if 'lidar_pseudo_img' not in inputs:
            raise ValueError("FusionBackbone requires 'lidar_pseudo_img' in inputs.")
        pc_img = inputs['lidar_pseudo_img']  # shape: [B, C, H, W] or [B, 1, H, W]
        if pc_img.ndim == 3:
            # If shape is [B, H, W], add channel dim
            pc_img = pc_img.unsqueeze(1)

        # Convert to float tensor if needed
        pc_img = torch.as_tensor(pc_img, dtype=torch.float32, device=next(self.parameters()).device)

        # 2) Encode LiDAR -> [B, dim]
        pc_feat = self.pc_encoder(pc_img)  # from your visual_model.py

        # 3) Encode camera, if using
        rgb_tokens = None
        if self.use_camera and 'img' in inputs:
            # Expect shape [B, T, 3, H, W] or [B, 3, H, W] (single frame).
            cam = inputs['img']
            if isinstance(cam, list):
                # In MMDet(3D), sometimes 'img' is a list of images per sample. 
                # You may need to stack or pick the first. This is dataset-specific.
                cam = cam[0]  # e.g., pick the first view
            cam = torch.as_tensor(cam, dtype=torch.float32, device=pc_img.device)
            if cam.ndim == 4:
                # => [B, 3, H, W], treat T=1
                cam = cam.unsqueeze(1)  # => [B, 1, 3, H, W]
            B, T, C, H, W = cam.shape
            # Flatten frames => [B*T, C, H, W]
            cam = cam.reshape(B*T, C, H, W)
            # Patch embedding
            cam_tokens = self.vit.patch_embed(cam)  # => [B*T, num_patches, dim]
            # Reshape => [B, T*num_patches, dim]
            cam_tokens = cam_tokens.reshape(B, -1, cam_tokens.shape[-1])
            # Prepend CLS token
            cls_token = self.vit.cls_token.expand(B, -1, -1)
            rgb_tokens = torch.cat([cls_token, cam_tokens], dim=1)

            # Add position embeddings
            pos_embed = self.vit.pos_embed  # => [1, 1+patch_count, dim]
            cls_pos   = pos_embed[:, 0:1, :]
            patch_pos = pos_embed[:, 1:, :]
            needed_patches = rgb_tokens.shape[1] - 1
            if patch_pos.shape[1] != needed_patches:
                # Interpolate
                patch_pos = nn.functional.interpolate(
                    patch_pos.transpose(1, 2),
                    size=needed_patches,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            rgb_tokens[:, 0:1, :] += cls_pos
            rgb_tokens[:, 1:, :]  += patch_pos
        else:
            # If no camera, we can still pass a dummy token, or just fuse PC alone
            # For demonstration, let's create a single token with zeros
            B = pc_feat.shape[0]
            dim = pc_feat.shape[1]
            rgb_tokens = torch.zeros((B, 1, dim), device=pc_feat.device, dtype=pc_feat.dtype)

        # 4) Pass through AdaptFormer blocks
        # pc_feat shape => [B, dim], turn it into [B, 1, dim]
        pc_feat = pc_feat.unsqueeze(1)
        for block in self.fusion_blocks:
            pc_feat, rgb_tokens = block(pc_feat, rgb_tokens)

        # 5) Post-norm
        pc_feat  = self.post_norm(pc_feat)      # => [B, 1, dim]
        rgb_tokens = self.post_norm(rgb_tokens) # => [B, N, dim]

        # 6) Extract "class tokens"
        pc_cls  = pc_feat[:, 0]  # => [B, dim]
        rgb_cls = rgb_tokens[:, 0]  # => [B, dim]
        fused   = 0.5 * (pc_cls + rgb_cls)

        return dict(fused_feat=fused)
