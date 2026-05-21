"""
CLIP Text Encoder for Open-Vocabulary classification.
Encodes class/predicate names into text embeddings for vision-language alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# Lazy import — avoid hard dependency on transformers at module level
_CLIPModel = None
_CLIPTokenizer = None
_transformers_available = None


def _ensure_transformers():
    """Import transformers lazily, caching the result."""
    global _CLIPModel, _CLIPTokenizer, _transformers_available
    if _transformers_available is not None:
        return _transformers_available
    try:
        # pyrefly: ignore [missing-import]
        from transformers import CLIPModel, CLIPTokenizer
        _CLIPModel = CLIPModel
        _CLIPTokenizer = CLIPTokenizer
        _transformers_available = True
    except ImportError:
        _transformers_available = False
        print("[CLIPTextEncoder] `transformers` library not installed. "
              "Install with: pip install transformers")
    return _transformers_available


class CLIPTextEncoder(nn.Module):
    """
    Frozen CLIP text encoder. Produces normalized text embeddings
    that live in the same space as the visual embeddings from ObjectHead / RelationHead.

    NOTE: The CLIP model is stored in self._clip_internals (a plain dict),
    NOT as a direct nn.Module attribute. This prevents PyTorch from
    registering its ~150M parameters as submodules, which would
    bloat state_dict / checkpoints and confuse the optimizer.
    """

    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32", clip_dim: int = 512):
        super().__init__()
        self.clip_model_name = clip_model_name
        self._clip_dim = clip_dim
        self._loaded = False

        # Store CLIP objects in a plain dict so PyTorch does NOT
        # register them as submodules (avoids huge state_dict).
        self._clip_internals: dict = {"model": None, "tokenizer": None}

        if not _ensure_transformers():
            return

        try:
            model = _CLIPModel.from_pretrained(clip_model_name)
            tokenizer = _CLIPTokenizer.from_pretrained(clip_model_name)
            self._clip_dim = model.config.projection_dim

            # Freeze all CLIP parameters
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

            self._clip_internals["model"] = model
            self._clip_internals["tokenizer"] = tokenizer
            self._loaded = True
        except Exception as e:
            print(f"[CLIPTextEncoder] Could not load CLIP model '{clip_model_name}': {e}. "
                  "Using random fallback.")

    @property
    def clip_dim(self) -> int:
        return self._clip_dim

    def _get_clip(self):
        return self._clip_internals["model"], self._clip_internals["tokenizer"]

    @torch.no_grad()
    def encode_text(self, class_names: List[str], device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Encode class names into L2-normalized text embeddings.

        Args:
            class_names: list of class name strings
            device: target device

        Returns:
            (num_classes, clip_dim) normalized embeddings
        """
        model, tokenizer = self._get_clip()

        if model is None or tokenizer is None:
            embeds = torch.randn(len(class_names), self._clip_dim)
            embeds = F.normalize(embeds, dim=-1)
            return embeds.to(device) if device else embeds

        prompts = [f"a photo of a {name}" for name in class_names]
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        if device:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model.to(device)

        text_features = model.get_text_features(**inputs)
        return F.normalize(text_features, dim=-1)

    @torch.no_grad()
    def encode_predicates(self, predicate_names: List[str], device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Encode predicate names for open-vocabulary relation detection.

        Uses an ensemble of prompt templates because CLIP was not trained
        on "a relation of X" syntax. Instead, we average embeddings from
        multiple natural-language templates:
          - "{name}"                          (bare word, e.g. "riding")
          - "a person {name} something"       (verb-form context)
          - "something {name} something"      (generic subject/object)

        This prompt ensemble improves alignment vs a single template.
        """
        model, tokenizer = self._get_clip()

        if model is None or tokenizer is None:
            embeds = torch.randn(len(predicate_names), self._clip_dim)
            embeds = F.normalize(embeds, dim=-1)
            return embeds.to(device) if device else embeds

        # Ensemble of prompt templates (averaged)
        templates = [
            "{}",                             # bare word: "riding"
            "a person {} something",          # verb context
            "something {} something",         # generic context
        ]

        all_embeds = []
        for template in templates:
            prompts = [template.format(name) for name in predicate_names]
            inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
            if device:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model.to(device)
            feats = model.get_text_features(**inputs)
            all_embeds.append(F.normalize(feats, dim=-1))

        # Average across templates, then re-normalize
        avg_embeds = torch.stack(all_embeds, dim=0).mean(dim=0)
        return F.normalize(avg_embeds, dim=-1)
