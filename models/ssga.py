import torch
import torch.nn as nn
import torch.nn.functional as F


class SSGA(nn.Module):
    """
    Sparse Scene-Graph-Guided Attention (SSGA).

    Allows relation information to flow back into object representations
    using a *sparse* attention mask derived from the predicted scene-graph
    topology (subject/object assignments from the Relation Head).

    Each object query only attends to relation queries where it is predicted
    as the subject or the object, making the attention graph-structured and
    sparse rather than fully dense.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # Projections for Q, K, V (manual, for custom masking)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._head_dim = d_model // nhead

        # Xavier Uniform init (consistent with Transformer — see §15.1)
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def _build_sparse_mask(
        self,
        sub_assignment: torch.Tensor,
        obj_assignment: torch.Tensor,
        num_obj: int,
        num_rel: int,
        top_k: int = 3,
    ) -> torch.Tensor:
        """
        Build a boolean adjacency mask (B, num_obj, num_rel) where
        mask[b, i, j] = True means object_i may attend to relation_j.

        A relation query j is linked to object i if i is among the top-k
        predicted subjects or objects for that relation.

        Args:
            sub_assignment: (B, num_rel, num_obj) raw logits / scores
            obj_assignment: (B, num_rel, num_obj) raw logits / scores
            num_obj: number of object queries
            num_rel: number of relation queries
            top_k: how many top subject/object indices per relation to keep

        Returns:
            mask: (B, num_obj, num_rel) boolean — True = allowed to attend
        """
        B = sub_assignment.shape[0]
        device = sub_assignment.device

        # Top-k subject / object indices per relation query
        top_k_actual = min(top_k, num_obj)
        sub_topk = sub_assignment.topk(top_k_actual, dim=-1).indices  # (B, num_rel, top_k)
        obj_topk = obj_assignment.topk(top_k_actual, dim=-1).indices

        mask = torch.zeros(B, num_obj, num_rel, dtype=torch.bool, device=device)
        for k_i in range(top_k_actual):
            # For each relation j, mark the k-th top subject as linked
            # sub_topk[:, :, k_i] -> (B, num_rel) indices into obj dimension
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_rel)
            rel_idx = torch.arange(num_rel, device=device).unsqueeze(0).expand(B, -1)

            mask[batch_idx, sub_topk[:, :, k_i], rel_idx] = True
            mask[batch_idx, obj_topk[:, :, k_i], rel_idx] = True

        return mask

    def forward(
        self,
        obj_queries: torch.Tensor,
        rel_queries: torch.Tensor,
        sub_assignment: torch.Tensor = None,
        obj_assignment: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            obj_queries:     (B, num_obj, d_model) — Q
            rel_queries:     (B, num_rel, d_model) — K, V
            sub_assignment:  (B, num_rel, num_obj) raw scores (optional; dense attn if None)
            obj_assignment:  (B, num_rel, num_obj) raw scores (optional)

        Returns:
            updated obj_queries: (B, num_obj, d_model)
        """
        B, num_obj, D = obj_queries.shape
        num_rel = rel_queries.shape[1]

        Q = self.q_proj(obj_queries)  # (B, num_obj, D)
        K = self.k_proj(rel_queries)  # (B, num_rel, D)
        V = self.v_proj(rel_queries)

        # Reshape for multi-head attention
        head_dim = self._head_dim
        nhead = self.nhead
        Q = Q.view(B, num_obj, nhead, head_dim).transpose(1, 2)  # (B, nhead, num_obj, head_dim)
        K = K.view(B, num_rel, nhead, head_dim).transpose(1, 2)
        V = V.view(B, num_rel, nhead, head_dim).transpose(1, 2)

        # Scaled dot-product attention scores
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)  # (B, nhead, num_obj, num_rel)

        # Apply sparse mask if available
        if sub_assignment is not None and obj_assignment is not None:
            sparse_mask = self._build_sparse_mask(
                sub_assignment.detach(), obj_assignment.detach(),
                num_obj, num_rel, top_k=3,
            )  # (B, num_obj, num_rel)
            # Expand mask for all heads
            sparse_mask = sparse_mask.unsqueeze(1).expand(-1, nhead, -1, -1)
            attn_weights = attn_weights.masked_fill(~sparse_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # handle all-masked rows
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # (B, nhead, num_obj, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, num_obj, D)
        attn_output = self.out_proj(attn_output)

        # Residual + LayerNorm
        obj_queries = self.norm(obj_queries + self.dropout(attn_output))
        return obj_queries
