import torch
import torch.nn as nn
from typing import Optional

from .masks import bucket_dim, build_head_mask_bias, parse_mask_spec, spec_can_empty_rows


class TransformerMask(nn.Transformer):
    """``nn.Transformer`` whose encoder is a :class:`MaskedEncoder`.

    The only argument is ``mask_spec`` for the encoder (see :class:`MaskedEncoder`).
    Everything else follows the standard ``nn.Transformer`` configuration: d_model=512,
    8 heads, 6 encoder and 6 decoder layers, feed-forward 2048, dropout 0.1, ReLU, and
    ``batch_first=False`` inputs of shape ``[seq, batch, d_model]``.
    With 8 heads, ``mask_spec`` is either one code or eight (e.g. ``"CCCCFFFF"``).

    The one departure from the defaults is ``norm_first=True`` (pre-norm) in both stacks.
    Pre-norm leaves the residual stream un-normalised between layers, so a token copied
    by attention reaches the output as itself; with post-norm the model escaped the
    answer-prior plateau on a needle-copy task but stalled short of solving it.

    The encoder is built exactly as ``nn.Transformer`` builds its own and passed in as
    ``custom_encoder``; the decoder, the forward pass and parameter initialisation are
    ``nn.Transformer``'s unchanged. The encoder ignores ``src_mask`` and
    ``src_is_causal`` because ``mask_spec`` decides its masking.
    """

    def __init__(self, mask_spec: str):
        d_model, nhead, num_encoder_layers = 512, 8, 6  # nn.Transformer's defaults
        # nn.TransformerEncoderLayer's remaining defaults (feed-forward 2048, dropout 0.1,
        # ReLU, eps 1e-5, bias) are the same as nn.Transformer's. norm_first is the one
        # departure, set on both stacks -- see the class docstring.
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, norm_first=True)
        encoder_norm = nn.LayerNorm(d_model)
        encoder = MaskedEncoder(encoder_layer, num_encoder_layers, mask_spec, encoder_norm)
        super().__init__(d_model, nhead, num_encoder_layers, custom_encoder=encoder,
                         norm_first=True)
        self.mask_spec = mask_spec


class MaskedEncoder(nn.TransformerEncoder):
    """``nn.TransformerEncoder`` that gives each self-attention head its own mask.

    ``mask_spec`` has one code per head -- ``"C"`` causal, ``"F"`` future-only -- so
    ``"CCCCFFFF"`` on an eight-head layer gives four causal and four future-only heads in
    every layer. A single character applies to every head. The masks come from
    :func:`masks.build_head_mask_bias`, so they are cached and use the finite
    ``MASK_NEG`` sentinel rather than ``-inf``.

    Args:
        encoder_layer: an ``nn.TransformerEncoderLayer``, cloned ``num_layers`` times.
        num_layers:    number of encoder layers.
        mask_spec:     per-head mask spec, e.g. ``"C"``, ``"F"`` or ``"CCCCFFFF"``.
        norm:          optional final normalisation.
    """

    def __init__(self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int,
                 mask_spec: str, norm: Optional[nn.Module] = None):
        super().__init__(encoder_layer, num_layers, norm=norm)
        self.num_heads = encoder_layer.self_attn.num_heads
        self.batch_first = encoder_layer.self_attn.batch_first
        parse_mask_spec(mask_spec, self.num_heads)  # reject a bad spec here, not on first forward
        self.mask_spec = mask_spec

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                is_causal: Optional[bool] = None) -> torch.Tensor:
        # mask and is_causal exist only because nn.Transformer.forward passes them.
        # mask_spec owns the attention mask, so a second one is refused rather than
        # silently dropped; is_causal is only a hint about that mask and is ignored.
        if mask is not None:
            raise ValueError("MaskedEncoder builds its mask from mask_spec; "
                             "pass mask=None (src_mask=None on TransformerMask).")
        if self.batch_first:
            bsz, seq_len = src.shape[0], src.shape[1]
        else:
            seq_len, bsz = src.shape[0], src.shape[1]
        # Build at the bucketed length and slice, never at the exact length. The mask
        # caches are keyed on length, and every eval prompt has its own, so exact-length
        # requests added a permanent entry per prompt -- ~2.5 GB each for CCCCFFFF at
        # 8192 -- until the GPU ran out. Every mask here depends only on j - i, so the
        # top-left L x L block of the bucketed plane is exactly the L x L mask.
        bias = build_head_mask_bias(self.mask_spec, self.num_heads, bucket_dim(seq_len),
                                    src.device, src.dtype)
        if bias is not None:
            bias = bias[..., :seq_len, :seq_len]
            if src_key_padding_mask is not None and spec_can_empty_rows(self.mask_spec,
                                                                        self.num_heads):
                # A future-only head at a padded query can only see later keys, which
                # are all padding too, so its row has nothing left to attend to.
                # PyTorch's eval fast path returns NaN for that row, and the next layer
                # carries it into real tokens. Unmasking padded-query rows is safe:
                # every real query masks the padded key columns, so nothing reads them.
                # Same fix as the old MaskedTransformer._build_attn_bias. masked_fill
                # (not in place) leaves the cached plane untouched.
                rows = src_key_padding_mask.to(torch.bool).view(bsz, 1, seq_len, 1)
                bias = bias.masked_fill(rows, 0.0)
            if bias.shape[:2] == (1, 1):
                # One code for every head and no per-sample rows: a 2-D mask broadcasts
                # over batch and heads.
                bias = bias[0, 0]
            else:
                # nn.MultiheadAttention takes per-head masks as [batch * heads, L, L],
                # batch-major, so row b * num_heads + h is head h of sample b.
                bias = bias.expand(bsz, self.num_heads, -1, -1).reshape(
                    bsz * self.num_heads, seq_len, seq_len)
        return super().forward(src, mask=bias,
                               src_key_padding_mask=src_key_padding_mask, is_causal=False)
