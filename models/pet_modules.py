import torch
import torch.nn as nn
import torch.nn.functional as F
import timm # Make sure timm is installed: pip install timm

# pet_modules.py (contents for AdaptFormer, as you provided)
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class AdaptFormer(nn.Module):
    """
    A simplified AdaptFormer block that:
      - Performs cross-attention between pc_tokens and rgb_tokens,
      - Uses a learnable set of latents to fuse them,
      - Has separate MLP down/up for each modality,
      - Applies separate scale params for attention and for the MLP residuals.
    """
    def __init__(self, num_latents, dim):
        super(AdaptFormer, self).__init__()

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1) # Consider making dropout rate configurable if needed
        self.dim = dim

        # --- Cross-attention scale parameters ---
        self.attn_scale_pc = nn.Parameter(torch.ones(1))   # scales cross-attn residual for pc
        self.attn_scale_rgb = nn.Parameter(torch.ones(1))  # scales cross-attn residual for rgb

        # --- MLP scale parameters ---
        self.mlp_scale_pc = nn.Parameter(torch.ones(1))    # scales MLP residual for pc
        self.mlp_scale_rgb = nn.Parameter(torch.ones(1))   # scales MLP residual for rgb

        # --- Down/Up Projection for Point Cloud tokens ---
        self.pc_down = nn.Linear(dim, dim)
        self.pc_up   = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.pc_down.weight)
        nn.init.zeros_(self.pc_down.bias)
        nn.init.xavier_uniform_(self.pc_up.weight)
        nn.init.zeros_(self.pc_up.bias)

        # --- Down/Up Projection for RGB tokens ---
        self.rgb_down = nn.Linear(dim, dim)
        self.rgb_up   = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.rgb_down.weight)
        nn.init.zeros_(self.rgb_down.bias)
        nn.init.xavier_uniform_(self.rgb_up.weight)
        nn.init.zeros_(self.rgb_up.bias)

        # --- Latents for cross-attention ---
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.empty(1, num_latents, dim).normal_(std=0.02))

        # Multi-head attention for cross-attention
        # Note: MHA default is batch_first=False, so inputs are (S, B, D)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8) # Removed batch_first=False as it's default

    def attention(self, q, k, v):
        """
        Wrapper for the PyTorch MultiheadAttention.
        Shape convention here is [S, B, D] for each of q, k, v.
        """
        # attn_output, attn_weights = self.attn(q, k, v, need_weights=False) # Optionally set need_weights
        attn_output, _ = self.attn(q, k, v) # If weights are not needed
        return attn_output  # [S, B, D]

    def fusion(self, pc_tokens, visual_tokens):
        """
        Cross-attention pipeline:
          1) Concat pc_tokens & visual_tokens -> combined.
          2) latents attend to combined -> fused_latents.
          3) pc_tokens attend to fused_latents.
          4) visual_tokens attend to fused_latents.
        """
        # pc_tokens: [B, 1, dim]
        # visual_tokens: [B, num_tokens, dim]

        B, pc_len, dim = pc_tokens.shape   # pc_len = 1
        _, vis_len, _  = visual_tokens.shape

        # (A) Combine pc + rgb along sequence dimension
        combined = torch.cat((pc_tokens, visual_tokens), dim=1)  # [B, 1 + vis_len, dim]
        # For multihead-attn, we want [seq_len, B, dim]
        combined = combined.permute(1, 0, 2)  # [1 + vis_len, B, dim]

        # (B) Expand latents for current batch
        # latents shape = [1, num_latents, dim], expand to [B, num_latents, dim]
        # then we want [num_latents, B, dim] for attn
        latents = self.latents.expand(B, -1, -1).permute(1, 0, 2)  # [num_latents, B, dim]

        # (C) latents attend to combined (cross-attention)
        fused_latents = self.attention(latents, combined, combined)  # [num_latents, B, dim]

        # (D) Now each modality attends to the fused latents
        # pc_tokens attends:
        pc_attn = self.attention(
            pc_tokens.permute(1, 0, 2),  # -> [1, B, dim]
            fused_latents,              # key
            fused_latents               # value
        )  # -> [1, B, dim]
        pc_attn = pc_attn.permute(1, 0, 2)  # back to [B, 1, dim]
        pc_tokens = pc_tokens + self.attn_scale_pc * pc_attn

        # visual_tokens attends:
        visual_attn = self.attention(
            visual_tokens.permute(1, 0, 2),  # -> [vis_len, B, dim]
            fused_latents,                  # key
            fused_latents                   # value
        )  # -> [vis_len, B, dim]
        visual_attn = visual_attn.permute(1, 0, 2)  # [B, vis_len, dim]
        visual_tokens = visual_tokens + self.attn_scale_rgb * visual_attn

        return pc_tokens, visual_tokens

    def forward_pc_AF(self, x):
        """
        MLP/Adapter for PC tokens: linear down -> activation -> dropout -> linear up.
        """
        x_down = self.pc_down(x)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up   = self.pc_up(x_down)
        return x_up

    def forward_visual_AF(self, x):
        """
        MLP/Adapter for RGB tokens: linear down -> activation -> dropout -> linear up.
        """
        x_down = self.rgb_down(x)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up   = self.rgb_up(x_down)
        return x_up

    def forward(self, pc, rgb):
        """
        pc:  [B, 1, dim]
        rgb: [B, num_tokens, dim]
        """
        # 1) Cross-attention / fusion
        pc_fused, rgb_fused = self.fusion(pc, rgb)

        # 2) MLP/Adapter with residual
        pc_out  = pc_fused  + self.mlp_scale_pc  * self.forward_pc_AF(pc_fused)
        rgb_out = rgb_fused + self.mlp_scale_rgb * self.forward_visual_AF(rgb_fused)

        return pc_out, rgb_out
# End of pet_modules.py content