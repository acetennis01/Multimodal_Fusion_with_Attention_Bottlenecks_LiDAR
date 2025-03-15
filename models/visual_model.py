# visual_model.py
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from models.pet_modules import AdaptFormer


class PointCloudEncoder(nn.Module):
    """
    Simple CNN-based encoder that takes a pseudo-image [B, 3, H, W]
    and produces a single [B, dim] descriptor.
    """
    def __init__(self, dim):
        super(PointCloudEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3,   64,  kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64,  128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        # After final pooling, we will adaptively pool to 14x14, then flatten -> 512 * 14 * 14.
        self.fc = nn.Linear(512 * 14 * 14, dim)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x shape: [B, 3, H, W]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)

        # Force 14x14 via adaptive pooling
        x = F.adaptive_avg_pool2d(x, (14, 14))    # [B, 512, 14, 14]
        x = x.flatten(start_dim=1)                # [B, 512*14*14]
        x = self.fc(x)                            # [B, dim]
        x = self.dropout(x)
        return x


class AVmodel(nn.Module):
    """
    Multimodal model that:
      - Uses a ViT patch embedding (from timm) for RGB,
      - Uses a CNN-based pseudo-image encoder for LiDAR,
      - Fuses them with a stack of AdaptFormer blocks,
      - Outputs classification logits.
    """
    def __init__(self, num_classes, num_latents, dim=768):
        super(AVmodel, self).__init__()

        # 1) RGB encoder (using timm's ViT patch embedding)
        self.v2 = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.v2.pre_logits = nn.Identity()
        self.v2.head       = nn.Identity()
        # We won't use self.v2.blocks here, because we do custom fusion.
        # But we keep self.v2.patch_embed, self.v2.cls_token, self.v2.pos_embed, etc.

        # 2) PC encoder
        self.pc_encoder = PointCloudEncoder(dim=dim)

        # 3) Stacked AdaptFormer blocks for cross-modality fusion
        encoder_layers = []
        for _ in range(12):
            encoder_layers.append(AdaptFormer(num_latents=num_latents, dim=dim))
        self.pointcloud_rgb_blocks = nn.Sequential(*encoder_layers)

        # 4) Final norm for both streams (reuse the ViT norm, which is LN)
        self.post_norm = self.v2.norm  # LayerNorm over dim=768

        # 5) Classifier head
        self.classifier = nn.Linear(dim, num_classes)

    def forward_pc_features(self, pc):
        """
        pc: [B, 3, H, W]
        Returns: [B, dim]
        """
        pc = self.pc_encoder(pc)
        return pc

    def forward_rgb_features(self, x):
        """
        x: [B, T, 3, H, W]
           (B = batch size, T = number of frames, 3 = channels, H/W = 224)
        Returns a token sequence of shape [B, 1 + (T*patches), dim].
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B, T, C, H, W), got {x.shape}")

        B, T, C, H, W = x.shape
        # Flatten frames into batch dimension
        x = x.reshape(B * T, C, H, W)  # => [B*T, 3, H, W]

        # Patch embedding
        # result shape: [B*T, num_patches, dim], typically [B*T, 196, 768] for 224x224 and patch16
        x = self.v2.patch_embed(x)

        # Reshape back to [B, T * num_patches, dim]
        x = x.reshape(B, -1, x.shape[-1])  # => [B, T*num_patches, dim]

        # Prepend CLS token
        cls_token = self.v2.cls_token.expand(B, -1, -1)   # => [B, 1, dim]
        x = torch.cat([cls_token, x], dim=1)              # => [B, 1 + T*num_patches, dim]

        # Split the official pos_embed into cls_pos + patch_pos
        # v2.pos_embed: [1, 1 + 196, dim] = [1, 197, dim] for base_patch16_224
        pos_embed   = self.v2.pos_embed  # shape: [1, 1 + patch_count, dim]
        cls_pos     = pos_embed[:, 0:1, :]   # => [1, 1, dim]
        patch_pos   = pos_embed[:, 1:, :]    # => [1, patch_count, dim]

        # If the user has T frames, total patches = T * patch_count_per_frame
        # We need to interpolate the patch portion accordingly.
        # The 'cls_pos' is never interpolated; it always remains one token.
        actual_patches = x.shape[1] - 1  # total patch tokens (T * patch_count_per_frame)
        if patch_pos.shape[1] != actual_patches:
            patch_pos = nn.functional.interpolate(
                patch_pos.transpose(1, 2),  # => [1, dim, patch_count]
                size=actual_patches,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # => [1, new_patch_count, dim]

        # Now add them
        x[:, 0:1, :]    = x[:, 0:1, :] + cls_pos  # add cls_pos
        x[:, 1:, :]     = x[:, 1:, :] + patch_pos

        return x  # shape: [B, 1 + T*patch_count, dim]

    def forward_encoder(self, pc, rgb):
        """
        pc:  [B, 1, dim]
        rgb: [B, N, dim]
        => Runs them through the 12 AdaptFormer blocks, then final LN, returns
           single token per stream => [B, dim], [B, dim]
        """
        # Expand pc to have "sequence length = 1"
        pc = pc.unsqueeze(1)  # => [B, 1, dim]

        # AdaptFormer blocks
        for blk in self.pointcloud_rgb_blocks:
            pc, rgb = blk(pc, rgb)

        # Post-norm
        pc  = self.post_norm(pc)   # => [B, 1, dim]
        rgb = self.post_norm(rgb)  # => [B, N, dim]

        # Extract the "class token" from each
        pc  = pc[:, 0]   # => [B, dim]
        rgb = rgb[:, 0]  # => [B, dim]

        return pc, rgb

    def forward(self, pc, rgb):
        """
        pc:  [B, 3, H, W]          (LiDAR pseudo-image)
        rgb: [B, T, 3, 224, 224]   (RGB frames or images)
        Returns classification logits: [B, num_classes].
        """

        # 1) Encode point cloud
        pc_feat = self.forward_pc_features(pc)  # => [B, dim]
        if torch.isnan(pc_feat).any():
            print("NaN in PC features")

        # 2) Encode RGB
        rgb_feat = self.forward_rgb_features(rgb)  # => [B, 1 + T*num_patches, dim]
        if torch.isnan(rgb_feat).any():
            print("NaN in RGB features")

        # 3) Fuse them
        pc_final, rgb_final = self.forward_encoder(pc_feat, rgb_feat)  # => [B, dim], [B, dim]

        # 4) Combine and classify
        fused = 0.5 * (pc_final + rgb_final)  # simple averaging
        logits = self.classifier(fused)       # => [B, num_classes]

        return logits
