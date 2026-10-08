"""Extract hidden states of one or more layers from a model on one input.

The hidden state of layer ``i`` is the residual stream of that layer right after its
attention has been added, BEFORE the LayerNorm and the feed-forward block that follow.
Every model here has exactly one LayerNorm sitting at that point, so the state is read
as that LayerNorm's input:

    transformer_mask / alibi, encoder layer    input of ``norm2``
    transformer_mask / alibi, decoder layer    input of ``norm3`` (after self- and cross-attention)
    roformer layer                     input of ``attention.output.LayerNorm``
"""

import torch

from ..models import DecoderOnlyLM


def _layers_and_norm(model, stack):
    """The layer list of *stack*, and the name of the LayerNorm the state is read at."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "decoder"):
        # transformer_mask, alibi (nn.Transformer). Reading the norm's INPUT is only "before
        # LayerNorm and FFN" when the layers are pre-norm, which they are.
        tr = model.transformer
        if stack == "encoder":
            return tr.encoder.layers, "norm2"
        return tr.decoder.layers, "norm3"
    if stack != "decoder":
        raise ValueError("this model is decoder-only; use stack='decoder'")
    return model.model.roformer.encoder.layer, "attention.output.LayerNorm"   # roformer


@torch.no_grad()
def extract_hidden_state(model, src, tgt_in, layer, stack="encoder"):
    """Return the hidden state of *layer* for one forward pass of ``model(src, tgt_in)``.

    The model is run in eval mode (and put back as it was), under bf16 autocast when the
    input is on a GPU, as prediction runs it.

    Args:
        model:  a model built by ``tmodel.models.build_model``, with trained weights loaded.
        src:    ``[batch, src_len]`` prompt ids.
        tgt_in: ``[batch, tgt_len]`` decoder input ids, starting with BOS. Pass BOS alone
                for the state at the first decoding step.
        layer:  layer index in the stack, 0-based; negative counts from the last. A
                sequence of indices reads all of them from the same forward pass.
        stack:  ``"encoder"`` or ``"decoder"``. The decoder-only model (roformer) has
                only the latter.

    Returns:
        ``[batch, seq, d_model]`` float32 on the CPU, or a list of them in the order
        given when *layer* is a sequence. ``seq`` is ``src_len`` for an
        encoder, ``tgt_len`` for its decoder, and ``src_len + tgt_len``
        for the decoder-only model, whose sequence is each prompt followed directly by
        its answer (``tmodel.models.pack_prompt_and_answer``).
    """
    layers, norm_name = _layers_and_norm(model, stack)
    single = isinstance(layer, int)
    indices = [layer] if single else list(layer)

    captured = [[] for _ in indices]
    handles = [layers[i].get_submodule(norm_name).register_forward_pre_hook(
                   lambda module, inputs, seen=seen: seen.append(inputs[0]))
               for i, seen in zip(indices, captured)]
    was_training = model.training
    model.eval()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=src.is_cuda):
            model(src, tgt_in)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    for i, seen in zip(indices, captured):
        if len(seen) != 1:
            raise RuntimeError(f"expected the hook on layer {i}'s {norm_name} to fire once, "
                               f"got {len(seen)}; the layer did not run its Python forward.")

    hidden = [seen[0] for seen in captured]
    if not isinstance(model, DecoderOnlyLM):
        hidden = [h.transpose(0, 1) for h in hidden]   # nn.Transformer is [seq, batch, d_model]
    hidden = [h.float().cpu() for h in hidden]
    return hidden[0] if single else hidden
