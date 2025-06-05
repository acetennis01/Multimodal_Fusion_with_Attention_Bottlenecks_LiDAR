# visual_model.py
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from models.pet_modules import AdaptFormer


class PointCloudEncoder(nn.Module):
    """
    Simple CNN-based encoder that takes a pseudo-image [B, 1, H, W]
    and produces a single [B, dim] descriptor.
    """
    def __init__(self, dim):
        super(PointCloudEncoder, self).__init__()

        self.conv1 = nn.Conv2d(1,   64,  kernel_size=3, stride=1, padding=1) # Changed: 3 to 1 input channel
        self.conv2 = nn.Conv2d(64,  128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        # After final pooling, we will adaptively pool to 14x14, then flatten -> 512 * 14 * 14.
        # Target H_out, W_out for adaptive pooling should match this calculation
        self.adaptive_pool_output_size = 14
        self.fc = nn.Linear(512 * self.adaptive_pool_output_size * self.adaptive_pool_output_size, dim)

        self.dropout = nn.Dropout(0.5) # Consider making dropout rate configurable

    def forward(self, x):
        # x shape: [B, 1, H, W]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2) # H/2, W/2

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2) # H/4, W/4

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2) # H/8, W/8

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2) # H/16, W/16

        # Force adaptive_pool_output_size x adaptive_pool_output_size (e.g., 14x14) via adaptive pooling
        x = F.adaptive_avg_pool2d(x, (self.adaptive_pool_output_size, self.adaptive_pool_output_size))
        x = x.flatten(start_dim=1)
        x = self.fc(x)
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
    def __init__(self, num_classes, num_latents, dim=768, vit_model_name='vit_base_patch16_224', adaptformer_blocks=12):
        super(AVmodel, self).__init__()
        self.dim = dim

        # 1) RGB encoder (using timm's ViT patch embedding)
        self.v2 = timm.create_model(vit_model_name, pretrained=True)
        # Modify ViT to only output features before the final head/block processing
        self.v2.head = nn.Identity() # Remove original head
        if hasattr(self.v2, 'pre_logits'): # Some ViT models have pre_logits
            self.v2.pre_logits = nn.Identity()
        
        # We won't use self.v2.blocks for main sequence processing, as we use custom fusion.
        # We need patch_embed, cls_token, pos_embed, and norm layer.
        if not hasattr(self.v2, 'patch_embed'):
            raise ValueError(f"ViT model {vit_model_name} does not have 'patch_embed' attribute.")
        if not hasattr(self.v2, 'pos_embed'):
            raise ValueError(f"ViT model {vit_model_name} does not have 'pos_embed' attribute.")
        if not hasattr(self.v2, 'cls_token'): # cls_token might be optional or handled differently in some ViTs
             # If no explicit CLS token, we might need to average pool patch tokens later or prepend a learned one.
             # For now, assuming standard ViT structure.
            print(f"Warning: ViT model {vit_model_name} might not have a 'cls_token'. Review RGB feature extraction.")


        # 2) PC encoder
        self.pc_encoder = PointCloudEncoder(dim=dim)

        # 3) Stacked AdaptFormer blocks for cross-modality fusion
        encoder_layers = []
        for _ in range(adaptformer_blocks): # Use parameter for number of blocks
            encoder_layers.append(AdaptFormer(num_latents=num_latents, dim=dim))
        self.pointcloud_rgb_blocks = nn.Sequential(*encoder_layers)

        # 4) Final norm for both streams (reuse the ViT norm, which is LayerNorm)
        if hasattr(self.v2, 'norm') and isinstance(self.v2.norm, nn.LayerNorm):
            self.post_norm = self.v2.norm
        else: # Fallback if ViT's norm layer isn't found or isn't LayerNorm
            print("Warning: Reusing ViT LayerNorm failed. Using a new LayerNorm.")
            self.post_norm = nn.LayerNorm(dim)


        # 5) Classifier head
        self.classifier = nn.Linear(dim, num_classes)

    def forward_pc_features(self, pc):
        """
        pc: [B, 1, H, W] (LiDAR pseudo-image)
        Returns: [B, dim]
        """
        pc_feat = self.pc_encoder(pc)
        return pc_feat

    def forward_rgb_features(self, x):
        """
        x: [B, T, 3, H, W]
           (B = batch size, T = number of frames, 3 = channels, H/W = 224 for ViT base)
        Returns a token sequence of shape [B, 1 + (T*patches_per_frame), dim].
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B, T, C, H, W), got {x.shape}")

        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)  # => [B*T, 3, H, W]

        # Patch embedding
        x = self.v2.patch_embed(x) # result shape: [B*T, num_patches_per_frame, dim]
                                   # e.g., [B*T, 196, 768] for 224x224 and patch16

        # Reshape back to [B, T * num_patches_per_frame, dim]
        num_patches_per_frame = x.shape[1]
        x = x.reshape(B, T * num_patches_per_frame, x.shape[-1])

        # Prepend CLS token
        if hasattr(self.v2, 'cls_token') and self.v2.cls_token is not None:
            cls_token = self.v2.cls_token.expand(B, -1, -1)   # => [B, 1, dim]
            x = torch.cat([cls_token, x], dim=1)              # => [B, 1 + T*num_patches_per_frame, dim]
            has_cls_token = True
        else: # Handle models without a CLS token (e.g., some Swin Transformers or if cls_token is None)
            has_cls_token = False
            # If no CLS token, the "class" representation will be derived differently,
            # e.g., by mean pooling patch tokens after encoder, or taking the 0-th token if it's learned differently.
            # The current `forward_encoder` takes rgb[:, 0], assuming a CLS token or equivalent is at index 0.

        # Positional Embedding
        # v2.pos_embed: [1, 1 + patches_per_frame, dim] for ViTs with CLS token
        # or [1, patches_per_frame, dim] for ViTs without CLS token (less common for base timm ViTs)
        pos_embed = self.v2.pos_embed

        if has_cls_token:
            cls_pos = pos_embed[:, 0:1, :]   # => [1, 1, dim]
            patch_pos_template = pos_embed[:, 1:, :]    # => [1, patches_per_frame_orig, dim]
            
            # Interpolate patch_pos_template if T > 1 or if patch_embed results in different patch count
            # target_num_patch_tokens = T * num_patches_per_frame
            target_num_patch_tokens = x.shape[1] - 1 # current number of patch tokens in x
            
            if patch_pos_template.shape[1] != target_num_patch_tokens:
                patch_pos_interpolated = F.interpolate(
                    patch_pos_template.transpose(1, 2),  # => [1, dim, patches_per_frame_orig]
                    size=target_num_patch_tokens,        # New size T * num_patches_per_frame
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)  # => [1, T*num_patches_per_frame, dim]
            else:
                patch_pos_interpolated = patch_pos_template

            x[:, 0:1, :] = x[:, 0:1, :] + cls_pos
            x[:, 1:, :] = x[:, 1:, :] + patch_pos_interpolated
        else: # No CLS token, apply pos_embed to all patch tokens
            target_num_patch_tokens = x.shape[1] # current number of patch tokens in x
            if pos_embed.shape[1] != target_num_patch_tokens:
                 pos_embed_interpolated = F.interpolate(
                    pos_embed.transpose(1, 2),
                    size=target_num_patch_tokens,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            else:
                pos_embed_interpolated = pos_embed
            x = x + pos_embed_interpolated
            
        return x

    def forward_encoder(self, pc_feat_single_token, rgb_feat_tokens):
        """
        pc_feat_single_token:  [B, dim] (from PointCloudEncoder)
        rgb_feat_tokens:       [B, N, dim] (N = 1 + T*num_patches_per_frame, from ViT features)
        
        Returns:
           pc_final_token: [B, dim]
           rgb_final_token: [B, dim] (typically the CLS token or equivalent from RGB stream)
        """
        # Expand pc_feat_single_token to have "sequence length = 1"
        pc_tokens = pc_feat_single_token.unsqueeze(1)  # => [B, 1, dim]

        # AdaptFormer blocks
        for blk in self.pointcloud_rgb_blocks:
            pc_tokens, rgb_feat_tokens = blk(pc_tokens, rgb_feat_tokens)

        # Post-norm
        pc_tokens  = self.post_norm(pc_tokens)   # => [B, 1, dim]
        rgb_feat_tokens = self.post_norm(rgb_feat_tokens)  # => [B, N, dim]

        # Extract the "class token" equivalent from each stream
        pc_final_token  = pc_tokens[:, 0]   # => [B, dim]
        rgb_final_token = rgb_feat_tokens[:, 0]  # Assumes the 0-th token is the CLS/summary token => [B, dim]

        return pc_final_token, rgb_final_token

    def forward(self, pc, rgb):
        """
        pc:  [B, 1, H, W]          (LiDAR pseudo-image)
        rgb: [B, T, 3, H_vit, W_vit] (RGB frames or images, e.g., H_vit=W_vit=224)
        Returns classification logits: [B, num_classes].
        """
        # 1) Encode point cloud
        pc_feat = self.forward_pc_features(pc)  # => [B, dim]
        # NaN check (optional, good for debugging)
        # if torch.isnan(pc_feat).any():
        #     print("NaN detected in PC features after pc_encoder")

        # 2) Encode RGB
        rgb_tokens = self.forward_rgb_features(rgb)  # => [B, 1 + T*num_patches_per_frame, dim]
        # NaN check (optional)
        # if torch.isnan(rgb_tokens).any():
        #     print("NaN detected in RGB features after ViT processing")

        # 3) Fuse them using AdaptFormer blocks
        pc_final, rgb_final = self.forward_encoder(pc_feat, rgb_tokens)  # => [B, dim], [B, dim]

        # 4) Combine and classify
        # Simple averaging, other strategies like concatenation + linear, or weighted sum could be used.
        fused = 0.5 * (pc_final + rgb_final)
        logits = self.classifier(fused)       # => [B, num_classes]

        return logits