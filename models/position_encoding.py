"""
Sinusoidal 2-D positional encoding (DETR-style).
Replaces the fixed learned 32x32 parameter that could not generalise across resolutions.
"""

import math
import torch
import torch.nn as nn


class PositionEmbeddingSine(nn.Module):
    """
    Sinusoidal positional encoding that adapts to any spatial resolution.
    Produces (B, d_model, H, W) positional embeddings.
    """

    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = True,
                 scale: float = None):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # half of d_model
        self.temperature = temperature
        self.normalize = normalize
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map (values are not used, only shape & device).
        Returns:
            pos: (B, num_pos_feats*2, H, W)
        """
        B, C, H, W = x.shape
        not_mask = torch.ones(B, H, W, dtype=torch.float32, device=x.device)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # (B, H, W, D)
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (B, 2*D, H, W)
        return pos
