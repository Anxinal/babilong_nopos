from typing import Optional

import torch
from torch import nn

from .attention import symmetric_alibi_bias
from .config import ALiBiConfig


def _per_head_bias(x: torch.Tensor, num_heads: int, batch_first: bool) -> torch.Tensor:
    """The symmetric ALiBi bias for *x*, as ``[batch, num_heads, seq, seq]``."""
    if batch_first:
        bsz, seq_len = x.shape[0], x.shape[1]
    else:
        seq_len, bsz = x.shape[0], x.shape[1]
    bias = symmetric_alibi_bias(num_heads, seq_len, x.device, x.dtype)
    return bias.unsqueeze(0).expand(bsz, -1, -1, -1)


class ALiBiEncoder(nn.TransformerEncoder):
    """``nn.TransformerEncoder`` with the symmetric ALiBi bias on every self-attention.

    The bias is passed as a float attention mask, which ``nn.MultiheadAttention`` adds
    to the scores before the softmax.
    """

    def __init__(self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int,
                 norm: Optional[nn.Module] = None):
        super().__init__(encoder_layer, num_layers, norm=norm)
        self.num_heads = encoder_layer.self_attn.num_heads
        self.batch_first = encoder_layer.self_attn.batch_first

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                is_causal: Optional[bool] = None) -> torch.Tensor:
        if mask is not None:
            raise ValueError("ALiBiEncoder builds its own attention bias; pass mask=None.")
        # nn.MultiheadAttention takes per-head masks as [batch * heads, L, L],
        # batch-major, so row b * num_heads + h is head h of sample b.
        bias = _per_head_bias(src, self.num_heads, self.batch_first).flatten(0, 1)
        return super().forward(src, mask=bias,
                               src_key_padding_mask=src_key_padding_mask, is_causal=False)


class ALiBiDecoder(nn.TransformerDecoder):
    """``nn.TransformerDecoder`` with the ALiBi bias on every self-attention.

    The bias is the encoder's symmetric one; ``tgt_mask`` (the causal mask) is applied
    on top, and below the diagonal the symmetric bias is the original causal ALiBi.
    Cross-attention gets no positional bias.
    """

    def __init__(self, decoder_layer: nn.TransformerDecoderLayer, num_layers: int,
                 norm: Optional[nn.Module] = None):
        super().__init__(decoder_layer, num_layers, norm=norm)
        self.num_heads = decoder_layer.self_attn.num_heads
        self.batch_first = decoder_layer.self_attn.batch_first

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False) -> torch.Tensor:
        bias = _per_head_bias(tgt, self.num_heads, self.batch_first)
        if tgt_mask is not None:
            if tgt_mask.dtype == torch.bool:
                bias = bias.masked_fill(tgt_mask, float("-inf"))   # True = blocked
            else:
                bias = bias + tgt_mask
        return super().forward(tgt, memory, tgt_mask=bias.flatten(0, 1),
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               tgt_is_causal=False, memory_is_causal=memory_is_causal)


class ALiBiTransformer(nn.Transformer):
    """``nn.Transformer`` (encoder-decoder) with ALiBi in place of a positional encoding.

    ``config`` sets the width, the heads, the layers per stack and the dropout; the rest
    is ``nn.Transformer``'s default (feed-forward 2048, ReLU, ``batch_first=False``).
    Both stacks are pre-norm, as in :class:`tmodel.maskedVanilla.TransformerMask`. The
    stacks are built as ``nn.Transformer`` builds its own and passed in as
    ``custom_encoder`` / ``custom_decoder``.
    """

    def __init__(self, config: ALiBiConfig) -> None:
        d_model, nhead = config.d_model, config.num_heads
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=config.dropout,
                                                   norm_first=True)
        encoder = ALiBiEncoder(encoder_layer, config.num_layers, nn.LayerNorm(d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dropout=config.dropout,
                                                   norm_first=True)
        decoder = ALiBiDecoder(decoder_layer, config.num_layers, nn.LayerNorm(d_model))
        super().__init__(d_model, nhead, config.num_layers, config.num_layers,
                         dropout=config.dropout, custom_encoder=encoder,
                         custom_decoder=decoder, norm_first=True)
