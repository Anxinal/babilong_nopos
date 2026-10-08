import math

import torch


def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2**math.floor(math.log2(n))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]


def symmetric_alibi_bias(num_heads: int, seq_len: int, device=None, dtype=None) -> torch.Tensor:
    """Symmetric ALiBi bias, ``[num_heads, seq_len, seq_len]``: ``-slope_h * |i - j|``.

    The "symmetric" encoder option of
    https://github.com/ofirpress/attention_with_linear_biases/issues/5, added to the
    attention scores before the softmax. Built at the length asked for rather than
    stored at a maximum length and sliced: the values are the same, and a stored
    ``[heads, maxpos, maxpos]`` plane would be gigabytes at the eval lengths.
    """
    context_position = torch.arange(seq_len, device=device)[:, None]
    memory_position = torch.arange(seq_len, device=device)[None, :]
    relative_position = memory_position - context_position
    relative_position = torch.abs(relative_position).unsqueeze(0).expand(num_heads, -1, -1)

    slopes = torch.tensor(get_slopes(num_heads), device=device, dtype=dtype) * -1
    return slopes.unsqueeze(1).unsqueeze(1) * relative_position
