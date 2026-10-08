"""maskedVanilla -- nn.Transformer with a per-head C/F encoder mask and no positional encoding.

    from tmodel.maskedVanilla import TransformerMask

    model = TransformerMask("CCCCFFFF")   # one code per head: B | C | F | S
"""

from .masks import (
    Mask,
    CausalMask,
    FutureOnlyMask,
    BidirectionalMask,
    build_additive_mask,
    build_head_mask_bias,
    parse_mask_spec,
    clear_mask_cache,
)
from .transformer_mask import TransformerMask, MaskedEncoder
