"""
Simple Feature Pyramid Network (FPN) for multi-scale feature fusion.
Fuses backbone outputs {C3, C4, C5} into enriched multi-scale features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FPN(nn.Module):
    """
    Top-down Feature Pyramid Network.

    Takes a list of multi-scale feature maps from the backbone and produces
    fused feature maps at each level via lateral connections and top-down pathway.
    """

    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_channels, 1))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

        # Initialise with kaiming
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: list of (B, C_i, H_i, W_i) from backbone (low → high level).
        Returns:
            list of (B, out_channels, H_i, W_i) fused feature maps.
        """
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="bilinear", align_corners=False
            )

        outs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        return outs
