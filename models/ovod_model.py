import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .transformer import build_transformer
from .prediction_heads import ObjectHead, RelationHead
from .ssga import SSGA
from .sgor import SGOR
from .fpn import FPN
from .position_encoding import PositionEmbeddingSine


class OVODModel(nn.Module):
    """
    End-to-end OVOD + OVRD model.

    Pipeline:
        backbone → FPN → input_proj → sinusoidal pos encoding
        → transformer encoder → decoder (with intermediate outputs)
        → for each decoder layer:
              split obj/rel queries → RelationHead (for SSGA mask)
              → SSGA (sparse scene-graph attention) → ObjectHead → SGOR
        → final outputs + auxiliary outputs for deep supervision
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = int(config.get("hidden_dim", 256))
        clip_dim = int(config.get("clip_dim", 512))
        num_obj_queries = int(config.get("num_obj_queries", 100))
        num_rel_queries = int(config.get("num_rel_queries", 50))
        num_classes = config.get("num_classes")
        num_rel_predicates = int(config.get("num_rel_predicates", 0) or 0)
        self.use_aux_loss = config.get("use_aux_loss", True)

        # --- Backbone ---
        self.backbone = build_backbone(config)

        # --- FPN for multi-scale fusion ---
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 640, 640)
            feats = self.backbone(dummy)
            in_channels_list = [f.shape[1] for f in feats]

        self.fpn = FPN(in_channels_list, hidden_dim)

        # Which FPN level to feed into the transformer (0=finest, -1=coarsest).
        # Coarsest keeps token count manageable for vanilla attention.
        self.fpn_level = int(config.get("fpn_level", -1))

        # 1x1 projection (identity when FPN already outputs hidden_dim,
        # kept for potential channel mismatch)
        self.input_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        # --- Sinusoidal positional encoding (DETR-style, resolution-agnostic) ---
        self.pos_encoding = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2)

        # --- Transformer ---
        self.transformer = build_transformer(config)

        # --- Learnable queries ---
        self.num_obj_queries = num_obj_queries
        self.num_rel_queries = num_rel_queries
        self.obj_query_embed = nn.Embedding(num_obj_queries, hidden_dim)
        self.rel_query_embed = nn.Embedding(num_rel_queries, hidden_dim)

        # --- Prediction heads ---
        self.obj_head = ObjectHead(hidden_dim, num_classes=num_classes, clip_dim=clip_dim)
        self.rel_head = RelationHead(
            hidden_dim,
            num_obj_queries=num_obj_queries,
            num_rel_predicates=num_rel_predicates,
            clip_dim=clip_dim,
        )

        # --- SSGA & SGOR ---
        self.ssga = SSGA(hidden_dim, int(config.get("nheads", 8)))
        self.sgor = SGOR(hidden_dim)

    def _split_queries(self, hs):
        """Split combined decoder output into object and relation features."""
        n_obj = self.num_obj_queries
        obj_feats = hs[:, :, :n_obj, :]   # (L, B, n_obj, D)
        rel_feats = hs[:, :, n_obj:, :]    # (L, B, n_rel, D)
        return obj_feats, rel_feats

    def _apply_heads(self, obj_feats, rel_feats, use_ssga=True):
        """
        Apply prediction heads to one decoder layer's output.
        Returns a dict of predictions.
        """
        # 1. Relation head first (needed for SSGA mask)
        pred_logits, sub_attn, obj_attn = self.rel_head(rel_feats, obj_feats)

        # 2. SSGA: sparse scene-graph attention to refine object features
        if use_ssga:
            obj_feats = self.ssga(
                obj_feats, rel_feats,
                sub_assignment=sub_attn,
                obj_assignment=obj_attn,
            )

        # 3. Object head: bbox + class
        bboxes, obj_logits = self.obj_head(obj_feats)

        # 4. SGOR: refine bounding boxes
        bboxes = self.sgor(obj_feats, bboxes)

        return {
            "pred_logits": obj_logits,
            "pred_boxes": bboxes,
            "rel_logits": pred_logits,
            "sub_assignment": sub_attn,
            "obj_assignment": obj_attn,
            "temperature": self.obj_head.temperature,
            "rel_temperature": self.rel_head.temperature,
        }

    def forward(self, samples: torch.Tensor):
        """
        Args:
            samples: (B, 3, H, W) normalised images

        Returns:
            dict with keys:
                pred_logits, pred_boxes, rel_logits, sub_assignment, obj_assignment
                aux_outputs (list of dicts, one per intermediate decoder layer)
        """
        # --- Backbone + FPN ---
        features = self.backbone(samples)
        fpn_features = self.fpn(features)
        src = self.input_proj(fpn_features[self.fpn_level])

        # --- Positional encoding (adapts to any resolution) ---
        pos = self.pos_encoding(src)

        # --- Prepare queries ---
        b = src.shape[0]
        obj_queries = self.obj_query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        rel_queries = self.rel_query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        combined_queries = torch.cat([obj_queries, rel_queries], dim=1)

        # --- Transformer ---
        hs, memory = self.transformer(src, None, combined_queries, pos)
        # hs: (num_decoder_layers, B, num_queries, D)

        # --- Split and apply heads ---
        obj_feats_all, rel_feats_all = self._split_queries(hs)
        num_layers = hs.shape[0]

        # Final layer: full pipeline (SSGA + SGOR)
        outputs = self._apply_heads(
            obj_feats_all[-1], rel_feats_all[-1], use_ssga=True,
        )

        # Auxiliary outputs from intermediate layers (no SSGA, lighter)
        if self.use_aux_loss and num_layers > 1:
            aux_outputs = []
            for l in range(num_layers - 1):
                aux = self._apply_heads(
                    obj_feats_all[l], rel_feats_all[l], use_ssga=False,
                )
                aux_outputs.append(aux)
            outputs["aux_outputs"] = aux_outputs

        return outputs


def build_model(config):
    return OVODModel(config)
