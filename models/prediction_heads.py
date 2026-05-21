import torch
import torch.nn as nn


class ObjectHead(nn.Module):
    """
    Bounding boxes (cx, cy, w, h) + class logits/embeddings.

    Two modes:
      - Closed-vocab: num_classes is set → logits are (num_classes + 1) including "no object".
      - Open-vocab:   num_classes is None → outputs live in clip_dim for contrastive alignment.

    A learnable temperature is included for scaling cosine similarities during
    open-vocabulary contrastive matching.
    """

    def __init__(self, d_model: int, num_classes: int = None, clip_dim: int = 512):
        super().__init__()
        self.num_classes = num_classes
        self.bbox_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
            nn.Sigmoid(),
        )

        if num_classes is not None:
            self.class_embed = nn.Linear(d_model, num_classes + 1)
        else:
            self.class_embed = nn.Linear(d_model, clip_dim)

        # Learnable temperature for contrastive similarity (used in open-vocab mode)
        self.log_temperature = nn.Parameter(torch.tensor(4.6052))  # ln(100) → τ ≈ 0.01

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, num_obj_queries, d_model)
        Returns:
            bboxes: (B, num_obj_queries, 4)
            logits: (B, num_obj_queries, num_classes+1) or (B, num_obj_queries, clip_dim)
        """
        bboxes = self.bbox_mlp(x)
        logits = self.class_embed(x)
        return bboxes, logits


class RelationHead(nn.Module):
    """
    Predicate logits/embeddings + subject/object pointer scores over object queries.

    Two modes:
      - Closed-vocab: num_rel_predicates > 0 → logits are (num_rel_predicates + 1).
      - Open-vocab:   → outputs live in clip_dim for contrastive alignment.
    """

    def __init__(self, d_model: int, num_obj_queries: int = 100,
                 num_rel_predicates: int = 0, clip_dim: int = 512):
        super().__init__()
        self.num_rel_predicates = num_rel_predicates
        if num_rel_predicates and num_rel_predicates > 0:
            self.predicate_embed = nn.Linear(d_model, num_rel_predicates + 1)
        else:
            self.predicate_embed = nn.Linear(d_model, clip_dim)

        self.sub_pointer = nn.Linear(d_model, d_model)
        self.obj_pointer = nn.Linear(d_model, d_model)

        # Learnable temperature for contrastive predicate matching
        self.log_temperature = nn.Parameter(torch.tensor(4.6052))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(self, rel_features: torch.Tensor, obj_features: torch.Tensor):
        """
        Args:
            rel_features: (B, num_rel_queries, d_model)
            obj_features: (B, num_obj_queries, d_model)
        Returns:
            pred_logits: (B, num_rel_queries, num_rel_predicates+1 or clip_dim)
            sub_attn:    (B, num_rel_queries, num_obj_queries)
            obj_attn:    (B, num_rel_queries, num_obj_queries)
        """
        pred_logits = self.predicate_embed(rel_features)

        sub_q = self.sub_pointer(rel_features)
        obj_q = self.obj_pointer(rel_features)

        # Scaled dot-product for pointer attention
        d = obj_features.shape[-1] ** 0.5
        sub_attn = torch.matmul(sub_q, obj_features.transpose(1, 2)) / d
        obj_attn = torch.matmul(obj_q, obj_features.transpose(1, 2)) / d

        return pred_logits, sub_attn, obj_attn
