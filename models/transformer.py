import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer


class CustomTransformerDecoder(nn.Module):
    """
    Transformer Decoder that returns intermediate outputs from each layer
    for auxiliary deep-supervision losses (DETR-style).
    """

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=True):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        intermediates = []

        for layer in self.layers:
            output = layer(
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            if self.return_intermediate:
                normed = self.norm(output) if self.norm else output
                intermediates.append(normed)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediates)  # (num_layers, B, N, C)
        return output.unsqueeze(0)  # (1, B, N, C)


class TransformerModule(nn.Module):
    """
    Transformer module containing Encoder and Decoder.
    Handles both Object Queries and Relation Queries.
    Returns intermediate decoder outputs for auxiliary losses.
    """

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.1, activation="relu",
                 return_intermediate=True):
        super().__init__()

        # Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        # Decoder (custom, returns intermediates)
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, batch_first=True
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = CustomTransformerDecoder(
            decoder_layer, num_decoder_layers, norm=decoder_norm,
            return_intermediate=return_intermediate,
        )

        self.d_model = d_model
        self.nhead = nhead
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        """
        Args:
            src:         (B, C, H, W) projected feature map
            mask:        (B, H*W) padding mask or None
            query_embed: (B, num_queries, C) combined Object + Relation queries
            pos_embed:   (B, C, H, W) positional embeddings

        Returns:
            hs:     (num_decoder_layers, B, num_queries, C) intermediate decoder outputs
            memory: (B, H*W, C) encoder output
        """
        B, C, H, W = src.shape
        src = src.flatten(2).transpose(1, 2)          # (B, N, C)
        pos_embed = pos_embed.flatten(2).transpose(1, 2)  # (B, N, C)

        # Encoder
        memory = self.encoder(src + pos_embed, src_key_padding_mask=mask)

        # Decoder: zero-init target, add query positional embeddings
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(
            tgt + query_embed, memory,
            memory_key_padding_mask=mask,
        )
        # hs: (num_layers, B, num_queries, C)
        return hs, memory


def build_transformer(config):
    return TransformerModule(
        d_model=config.get("hidden_dim", 256),
        nhead=config.get("nheads", 8),
        num_encoder_layers=config.get("enc_layers", 6),
        num_decoder_layers=config.get("dec_layers", 6),
        dim_feedforward=config.get("dim_feedforward", 1024),
        dropout=config.get("dropout", 0.1),
        return_intermediate=config.get("use_aux_loss", True),
    )
