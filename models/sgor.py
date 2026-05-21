import torch
import torch.nn as nn


class SGOR(nn.Module):
    """
    Scene-Graph-Based Offset Regression (SGOR).

    Refines bounding boxes by predicting residual offsets.
    Designed to be applied iteratively at each decoder layer for
    progressive box refinement (similar to Deformable-DETR iterative refinement).

    Uses inverse-sigmoid / sigmoid parameterisation for numerical stability
    when operating on normalised coordinates.
    """

    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # (dx, dy, dw, dh)
        )
        # Initialise last layer with small weights so initial offsets ≈ 0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        x = x.clamp(eps, 1.0 - eps)
        return torch.log(x / (1.0 - x))

    def forward(self, obj_features: torch.Tensor, current_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obj_features:  (B, num_obj_queries, d_model)
            current_boxes: (B, num_obj_queries, 4) — [cx, cy, w, h] normalised [0,1]

        Returns:
            refined_boxes: (B, num_obj_queries, 4) — updated normalised boxes
        """
        # Work in inverse-sigmoid space for stability
        boxes_inv = self._inverse_sigmoid(current_boxes.detach())

        combined = torch.cat([obj_features, boxes_inv], dim=-1)
        offsets = self.mlp(combined)

        # Refine in inverse-sigmoid space, then back to [0,1]
        refined = (boxes_inv + offsets).sigmoid()
        return refined
