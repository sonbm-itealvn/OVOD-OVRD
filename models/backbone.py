import torch
import torch.nn as nn
import timm


class Backbone(nn.Module):
    """
    Visual backbone: multi-scale feature maps via timm (Swin, ResNet, ...).
    """

    def __init__(self, name="swin_tiny_patch4_window7_224", pretrained=True, out_indices=(1, 2, 3)):
        super().__init__()
        self.name = name
        if "resnet" in name.lower():
            out_indices = out_indices if out_indices else (1, 2, 3)
        kwargs = dict(pretrained=pretrained, features_only=True, out_indices=out_indices)
        if "swin" in name.lower():
            kwargs["strict_img_size"] = False
        self.model = timm.create_model(name, **kwargs)
        self.out_channels = list(self.model.feature_info.channels())

    def forward(self, x):
        features = self.model(x)
        out = []
        for i, f in enumerate(features):
            if f.dim() == 4:
                c = self.out_channels[i]
                if f.shape[1] != c and f.shape[-1] == c:
                    f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
        return out


def build_backbone(config):
    name = config.get("backbone") or config.get("backbone_name", "swin_tiny_patch4_window7_224")
    return Backbone(
        name=name,
        pretrained=config.get("pretrained", True),
        out_indices=tuple(config.get("out_indices", (1, 2, 3))),
    )
