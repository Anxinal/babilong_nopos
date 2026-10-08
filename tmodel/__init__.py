"""tmodel -- the models compared in the positional-encoding vs attention-mask experiments.

Three models, built for training by :func:`tmodel.models.build_model`:

* ``transformer_mask`` -- :class:`TransformerMask`, ``nn.Transformer`` (encoder-decoder)
  (``tmodel.maskedVanilla``) whose encoder heads each get a C/F mask from ``mask_spec``. No positional encoding.
* ``roformer`` -- ``tmodel.roformer.RoFormerForCausalLM`` (decoder-only, RoPE).
* ``alibi``    -- :class:`ALiBiTransformer`, ``nn.Transformer`` (encoder-decoder, 4 + 4
  layers) with the symmetric ALiBi attention bias. No positional encoding.

RoFormer is not imported here: it needs ``transformers>=5``, and importing it eagerly
would make every model depend on that. Import it from ``tmodel.roformer`` directly.
"""

from .maskedVanilla import TransformerMask, MaskedEncoder
from .alibi import ALiBiConfig, ALiBiTransformer
