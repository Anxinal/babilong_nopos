
import math

import torch
from abc import ABC, abstractmethod


def fill_with_neg_inf(t: torch.Tensor) -> torch.Tensor:
    """Fill a tensor with -inf (in-place) and return it."""
    return t.fill_(float("-inf"))


# Defaults for the soft causal mask ("S"). Defined here as the single source of truth:
# train.py's argparse defaults and model_wrappers.py's checkpoint fallbacks both import
# these, so the value used at training and the value used at evaluation cannot drift.
# That matters more than usual here -- the mask is neither a parameter nor a buffer, so
# load_state_dict cannot detect a mismatch and a wrong cap would be evaluated silently.
#
# cap is in units of the ALREADY-SCALED attention logits: SDPA computes
# softmax(QK^T / sqrt(head_dim) + attn_mask), so the bias lands after the scaling, the
# same as the ALiBi bias in PositionalEmbeddings.py. A fully penalised future key gets
# exp(-cap) of the softmax weight an equal unpenalised past key gets, so total far-future
# mass is bounded by n_keys * exp(-cap). At cap=12 that is 1.3% over 2048 keys and 5.0%
# over 8192 -- chosen for the longest EVALUATION length rather than the training length,
# since a cap tuned at 2048 would leak four times as much at the top of the eval ladder
# and the mask would itself become a length-generalisation failure. It also sits ~5000x
# below fp16's 65504, so the cast in build_additive_mask is never near its range.
SOFT_MASK_CAP_DEFAULT = 12.0
# tau is an absolute token count, never a fraction of the sequence length: the bias
# depends only on the offset j - i, which is what makes it identical at 2048 and 8192.
# At cap=12 the near-diagonal slope is cap/tau = 0.19 logits per token, inside the range
# of ALiBi's eight-head schedule (0.5 ... 0.0039) already used in this repo.
SOFT_MASK_TAU_DEFAULT = 64.0

# Value written at masked positions, in place of -inf OR the dtype minimum.
#
# It has to satisfy two things at once, and torch.finfo(dtype).min satisfies only the
# first:
#
#   1. exp(MASK_NEG - row_max) must underflow to exactly 0, or masking leaks. At -1e4
#      that holds in fp16, bf16 and fp32 alike -- fp32 exp() already underflows below
#      about -104.
#   2. MASK_NEG + logit must stay INSIDE the dtype's range. fp16 saturates at 65504, so
#      finfo(float16).min = -65504 overflows to -inf as soon as an attention logit is
#      more than ~16 below zero at a masked position. -1e4 leaves 6x headroom.
#
# Why that second point matters here: the masked arms (C / F / CCCCFFFF) carry this value
# across roughly half of every [L, L] score matrix -- ~2M entries per head at L=2048 --
# while the bidirectional arm carries it only in padded key columns, a few dozen columns.
# So the masked arms have millions of chances per batch to overflow where BBBBBBBB has
# thousands, and in practice every masked arm collapsed in fp16 within ~6 epochs while
# BBBBBBBB trained cleanly past 15. That is the one structural difference between them.
#
# Trading the dtype minimum for a bounded sentinel costs nothing measurable: exp() gives
# 0 either way, so the attention weights are identical, and the tensor no longer sits at
# the edge of the representable range where a single addition leaves it.
MASK_NEG = -1e4


def bucket_dim(dim: int) -> int:
    """Round a sequence length up to the next power of two (minimum 256).

    Both mask caches are keyed on the sequence length, and at prediction time
    ``--batch_size 1`` means that length is each prompt's exact token count. Eval prompts
    do not come out all the same length -- vt_2chain at the 2048 rung spans 1913-2048 --
    so every prompt was minting a fresh cache entry that nothing ever reused. For a
    per-head spec such as CCCCFFFF the entry is ``[1, heads, L, L]``: 2.00 GB at L=8192
    in float32, so roughly forty distinct lengths exhausted an 80 GB card. That is the
    8192 OOM, and it is confined to masked arms because an all-bidirectional spec returns
    None and caches nothing at all.

    Bucketing is sound because every mask here is a function of the offset ``j - i``
    alone, so the top-left ``q x q`` block of a ``D x D`` plane IS the ``q x q`` plane.
    The callers already slice with ``mask[..., :q_len, :k_len]``, so building at the
    bucketed size and slicing returns exactly what building at the exact size would.

    Powers of two, rather than a finer grid, because entries below the largest one in use
    are redundant once it exists: a single 8192 plane serves every prompt from 4097 up.
    """
    return 1 << max(8, (max(int(dim), 1) - 1).bit_length())




class Mask(ABC):
    """Base class for a square additive attention mask.

    Subclasses implement ``_build`` to return a ``[dim, dim]`` float tensor that is added
    to the attention logits: ``0`` leaves a position untouched and a negative entry
    down-weights it. The hard masks use only ``0`` and ``-inf``, but an entry may be any
    finite negative value -- :class:`SoftCausalMask` grades its penalty with distance --
    so do not assume the tensor is two-valued. ``apply`` adds the mask to an
    attention-weight tensor in place of the standard ``attn_weights += attn_mask``
    pattern used throughout fairseq.
    """

    def __init__(self, dim: int):
        self._mask = self._build(dim)

    @abstractmethod
    def _build(self, dim: int) -> torch.Tensor: ...

    @property
    def tensor(self) -> torch.Tensor:
        """The raw ``[dim, dim]`` additive mask tensor.

        ``0`` = attend unpenalised, negative = down-weighted, ``-inf`` = blocked.
        """
        return self._mask

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Add the mask to *x*, broadcasting over any leading batch/head dims."""
        return x + self._mask.to(x)

    @classmethod
    def convert_from_config(cls, config: str):
        """Return the Mask **subclass** for a one-letter config code.

        ``"B"`` → BidirectionalMask, ``"C"`` → CausalMask, ``"F"`` → FutureOnlyMask,
        ``"S"`` → SoftCausalMask. The caller is responsible for instantiating the
        returned class; note ``SoftCausalMask`` takes ``cap``/``tau`` beyond ``dim``.

        Raises:
            ValueError: on an unrecognised code.

        This used to fall through to ``BidirectionalMask`` for anything unknown, which
        meant a code added to :data:`VALID_MASK_CODES` but forgotten here would silently
        train a *bidirectional* model while reporting the new code in its config -- a
        null result indistinguishable from a real one. Raising is safe: the only caller
        is :func:`build_additive_mask`, reached after ``parse_mask_spec`` has already
        validated the code, so this can only fire on a direct call.
        """
        if config == "B":
            return BidirectionalMask
        elif config == "C":
            return CausalMask
        elif config == "F":
            return FutureOnlyMask
        elif config == "S":
            return SoftCausalMask
        raise ValueError(
            f"Unknown mask code {config!r}; valid codes are {'/'.join(VALID_MASK_CODES)}."
        )

class CausalMask(Mask):
    """Standard autoregressive (causal) mask.

    Each token may attend to itself and all *earlier* tokens; strictly future
    positions (j > i) are set to -inf. The diagonal is 0, so self-attention is
    always allowed -- see the example below, and note that a mask blocking j >= i
    would leave the first row entirely masked.

    Example (dim=4):
        [  0  -∞  -∞  -∞ ]
        [  0   0  -∞  -∞ ]
        [  0   0   0  -∞ ]
        [  0   0   0   0 ]
    """

    def _build(self, dim: int) -> torch.Tensor:
        return torch.triu(fill_with_neg_inf(torch.zeros(dim, dim)), 1)


class FutureOnlyMask(Mask):
    """Anti-causal (future-only) mask.

    Each token may attend to itself and all *later* tokens; past
    positions (j < i) are set to -inf.  Used in encoder self-attention
    experiments — it must not be applied to the decoder, where it would
    break the autoregressive property.

    Args:
        dim:        Sequence length (mask will be ``[dim, dim]``).
        allow_self: If ``True`` (default) the diagonal is kept at 0 so
                    each token can still attend to itself.  If ``False``
                    the diagonal is also masked, which produces empty
                    attention rows for the last token and should be used
                    with care.

    """

    def __init__(self, dim: int, allow_self: bool = True):
        self.allow_self = allow_self
        super().__init__(dim)

    def _build(self, dim: int) -> torch.Tensor:
        diagonal = -1 if self.allow_self else 0
        return torch.tril(fill_with_neg_inf(torch.zeros(dim, dim)), diagonal)


class BidirectionalMask(Mask):
    """No-op mask: every position may attend to every other position.

    Equivalent to passing ``attn_mask=None`` but lets all three mask types
    be handled uniformly in per-head mask specs.
    """

    def _build(self, dim: int) -> torch.Tensor:
        return torch.zeros(dim, dim)


class SoftCausalMask(Mask):
    """Causal mask with a graded, bounded penalty instead of a hard block.

    The past is free and the future is discouraged by an amount that grows with distance
    and saturates. For query ``i`` and key ``j``, with ``d = j - i``:

    * ``j < i``  (past):            bias ``0``
    * ``j >= i`` (present/future):  bias ``-cap * tanh(d / tau)``

    ``f(0) = 0``, so the diagonal is never penalised and self-attention is always free;
    ``f`` is strictly increasing in ``d`` and bounded above by ``cap``. A single ``f`` is
    shared by every head, unlike ALiBi's per-head slope schedule.

    This gives a continuous knob between the two hard masks that none of ``B``/``C``/``F``
    provides: ``cap = 0`` is exactly bidirectional, and a large ``cap`` approaches causal.

    Example (dim=4, cap=12, tau=2, rounded):
        [  0.00  -5.57  -9.17 -11.07 ]
        [  0.00   0.00  -5.57  -9.17 ]
        [  0.00   0.00   0.00  -5.57 ]
        [  0.00   0.00   0.00   0.00 ]

    Args:
        dim: sequence length (mask will be ``[dim, dim]``).
        cap: upper bound on the penalty, in units of the **already-scaled** attention
            logits -- SDPA applies ``1 / sqrt(head_dim)`` itself, so this bias is added
            after that scaling. A fully penalised future key receives ``exp(-cap)`` of
            the softmax weight an equal unpenalised past key receives.
        tau: distance scale, an absolute token count. ``tanh`` reaches 0.76 of ``cap`` at
            ``d = tau`` and 0.995 at ``d = 3 * tau``; near the diagonal the penalty grows
            at ``cap / tau`` logits per token.

    Raises:
        ValueError: on a non-positive ``tau`` (which would make ``0 / 0`` a NaN on the
            diagonal, surfacing only as a NaN loss much later), a negative ``cap`` (which
            would *reward* the future), or a non-finite/absurd ``cap``.
    """

    def __init__(self, dim: int,
                 cap: float = SOFT_MASK_CAP_DEFAULT,
                 tau: float = SOFT_MASK_TAU_DEFAULT):
        cap = float(cap)
        tau = float(tau)
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError(
                f"SoftCausalMask: tau must be finite and strictly positive, got {tau!r}. "
                f"tau=0 would make d/tau a NaN on the diagonal."
            )
        if not math.isfinite(cap) or cap < 0.0:
            raise ValueError(
                f"SoftCausalMask: cap must be finite and non-negative, got {cap!r}. "
                f"A negative cap would reward attending to the future; use cap=0 for a "
                f"bidirectional mask."
            )
        if cap > 1e4:
            raise ValueError(
                f"SoftCausalMask: cap={cap!r} is far beyond anything meaningful -- "
                f"exp(-100) already underflows float32, so the mask is indistinguishable "
                f"from hard causal well below this, and fp16 saturates at 65504."
            )
        self.cap = cap
        self.tau = tau
        super().__init__(dim)  # must come last: the base ctor calls _build immediately

    def _build(self, dim: int) -> torch.Tensor:
        pos = torch.arange(dim)
        # j - i, floored at 0 so the past half-plane and the diagonal are both exactly
        # zero in one step (d=0 -> tanh(0) = 0). Branch-free, so f(0)=0, strict
        # monotonicity and boundedness all follow from the closed form.
        d = (pos.unsqueeze(0) - pos.unsqueeze(1)).clamp_min(0).to(torch.float32)
        return -self.cap * torch.tanh(d / self.tau)


# ---------------------------------------------------------------------------
# Cached additive masks
# ---------------------------------------------------------------------------

_MASK_CACHE = {}


def build_additive_mask(mask_type: str, dim: int, device, dtype, *,
                        soft_cap: float = SOFT_MASK_CAP_DEFAULT,
                        soft_tau: float = SOFT_MASK_TAU_DEFAULT) -> torch.Tensor:
    """Return a cached ``[dim, dim]`` additive mask, or ``None`` for bidirectional.

    Two differences from instantiating a :class:`Mask` directly, both of which matter
    in the training/inference hot path:

    * **Cached.** Building the tensor fresh on every forward pass costs a 268 MB
      allocation at ``dim=8192``. The cache is keyed on shape, device and dtype, and
      the number of distinct keys is bounded by the experiment grid.
    * **Finite and bounded.** Masked positions hold :data:`MASK_NEG` rather than ``-inf``
      or the dtype minimum. Finiteness stops a fully masked row -- which happens with the
      future-only mask when a padded query can only see later positions that are
      themselves padding -- from softmaxing to NaN; such a NaN is not harmless, because
      the decoder's cross-attention multiplies it by a zero weight and ``0 * NaN`` is
      NaN, so it reaches real positions. Boundedness stops ``MASK_NEG + logit`` from
      overflowing fp16, which the dtype minimum does not. See :data:`MASK_NEG`.
      ``"S"`` never needs this: its penalty is bounded by ``soft_cap``, so no row can be
      fully masked and the post-processing below is a no-op on it.

    Args:
        mask_type: ``"B"`` bidirectional, ``"C"`` causal, ``"F"`` future-only,
                   ``"S"`` soft causal (graded finite penalty on the future).
        dim:       sequence length.
        device:    target device.
        dtype:     target floating dtype (must match the attention query dtype).
        soft_cap:  ``"S"`` only -- penalty ceiling, in already-scaled logit units.
        soft_tau:  ``"S"`` only -- distance scale in tokens. See :class:`SoftCausalMask`.

    Returns:
        ``[dim, dim]`` additive mask, or ``None`` when ``mask_type`` is ``"B"``.
        Note ``"S"`` never returns ``None``, even at ``soft_cap == 0``; callers that want
        that degeneracy rewrite the code to ``"B"`` instead (see
        :func:`build_head_mask_bias`), because the homogeneous path there dereferences
        the returned plane without a ``None`` check.
    """
    if mask_type == "B":
        return None  # bidirectional: nothing to add

    # The soft mask's parameters are part of its identity, so two models differing only
    # in cap or tau must not share a cached plane. They are appended only for "S" so that
    # every existing key stays byte-identical -- which preserves the property documented
    # in build_head_mask_bias that the decoder's causal plane is literally the same object
    # as the encoder's, even when the encoder is sweeping a non-default cap.
    key = (mask_type, dim, str(device), dtype)
    if mask_type == "S":
        key += (float(soft_cap), float(soft_tau))
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        return cached

    mask_cls = Mask.convert_from_config(mask_type)
    # convert_from_config returns a class, which is what every other branch wants; only
    # the soft mask needs constructor arguments beyond dim.
    if mask_type == "S":
        tensor = mask_cls(dim, cap=soft_cap, tau=soft_tau).tensor
    else:
        tensor = mask_cls(dim).tensor  # float32, -inf in masked positions
    # MASK_NEG, not finfo(dtype).min -- see the note on MASK_NEG. nan_to_num replaces the
    # -inf the Mask subclasses produce; the clamp then catches anything the soft mask or a
    # future subclass might put below the sentinel.
    tensor = torch.nan_to_num(tensor, neginf=MASK_NEG).to(device=device, dtype=dtype)
    tensor = tensor.clamp_min(MASK_NEG)

    _MASK_CACHE[key] = tensor
    return tensor


def clear_mask_cache() -> None:
    """Drop every cached mask. Mainly useful in tests and to release device memory."""
    _MASK_CACHE.clear()
    _HEAD_MASK_CACHE.clear()


# ---------------------------------------------------------------------------
# Per-head mask specs
# ---------------------------------------------------------------------------

VALID_MASK_CODES = ("B", "C", "F", "S")

# Codes whose mask can leave a query row with no attendable key at all, once the key
# padding mask is folded in on top of it.
#
# Only "F" can. A future-only head at query i keeps j >= i, so for a query sitting in
# the padded tail of a ragged batch every key it is allowed to see is padding, and the
# combined bias for that row is uniformly -inf/neg. "C" and "B" cannot: whatever the
# padding, a causal or bidirectional row always retains at least one real key at j <= i
# (position 0 is never padding, since padding is a right-hand tail). "S" cannot either,
# because its penalty is bounded by soft_cap and never blocks anything outright.
#
# A fully masked row is not merely meaningless, it is a NaN source: the memory-efficient
# SDPA backend returns NaN for such rows even when the mask is the dtype minimum rather
# than -inf (pytorch/pytorch#110213 -- the reason transformers carries
# AttentionMaskConverter._unmask_unattended). The NaN then leaves the encoder at a padded
# position and reaches every real position through the decoder's cross-attention, where
# the padded key is multiplied by a zero weight and 0 * NaN = NaN.
#
# MaskedTransformer._build_attn_bias consults this to decide whether the rows belonging
# to padded queries need neutralising. See the note there.
EMPTY_ROW_CODES = ("F",)


def spec_can_empty_rows(spec: str, num_heads: int) -> bool:
    """Whether *spec* can produce a query row with nothing left to attend to.

    True only when the spec contains a code from :data:`EMPTY_ROW_CODES`, and only
    relevant for a batch that actually carries key padding -- without padding, every
    code including ``"F"`` keeps at least the diagonal.

    Args:
        spec:      per-head mask spec, e.g. ``"CCCCFFFF"``.
        num_heads: number of attention heads (for expanding a single-code spec).
    """
    return any(code in EMPTY_ROW_CODES for code in parse_mask_spec(spec, num_heads))

_HEAD_MASK_CACHE = {}


def parse_mask_spec(spec: str, num_heads: int) -> list:
    """Expand a per-head mask spec into one code per head.

    A spec assigns a mask to each attention head, so ``"CCCCFFFF"`` on an eight-head
    model gives four causal heads and four future-only heads in every layer that uses
    it. A single character is shorthand for every head, which keeps ``"C"`` meaning
    ``"CCCCCCCC"``.

    Head order carries no meaning. Heads are concatenated and mixed by one output
    projection, so ``"CCCCFFFF"`` and ``"CFCFCFCF"`` describe the same model up to a
    permutation of that projection's input; only the count of each code matters. The
    spec is honoured as written regardless.

    Args:
        spec:      one code per head, or a single code for all of them.
        num_heads: number of attention heads.

    Returns:
        List of ``num_heads`` single-character codes.

    Raises:
        ValueError: on an unknown code or a length that is neither 1 nor *num_heads*.
    """
    cleaned = spec.strip().upper()
    if not cleaned:
        raise ValueError("Mask spec is empty; expected codes from " + "/".join(VALID_MASK_CODES))

    unknown = sorted(set(cleaned) - set(VALID_MASK_CODES))
    if unknown:
        raise ValueError(
            f"Mask spec {spec!r} contains unknown code(s) {''.join(unknown)!r}; "
            f"valid codes are {'/'.join(VALID_MASK_CODES)}."
        )

    if len(cleaned) == 1:
        return [cleaned] * num_heads
    if len(cleaned) != num_heads:
        raise ValueError(
            f"Mask spec {spec!r} has length {len(cleaned)} but the model has "
            f"{num_heads} heads. Give one code per head, or a single code for all."
        )
    return list(cleaned)


def build_head_mask_bias(spec: str, num_heads: int, dim: int, device, dtype, *,
                         soft_cap: float = SOFT_MASK_CAP_DEFAULT,
                         soft_tau: float = SOFT_MASK_TAU_DEFAULT):
    """Return the additive mask for a per-head spec, cached and reused.

    Three cases, in increasing cost:

    * every head bidirectional -> ``None``, nothing is allocated
    * one code for every head  -> ``[1, 1, dim, dim]`` view of the plane that
      :func:`build_additive_mask` already caches, so no memory is duplicated and the
      decoder's causal mask is literally the same object
    * mixed codes -> ``[1, num_heads, dim, dim]``

    Only the mixed case duplicates anything. A single attention call takes exactly one
    mask tensor, so two different planes can only reach two different heads by sitting
    together in one contiguous tensor. That tensor is assembled once from the cached
    planes and then cached itself, so the duplication is resident memory rather than
    repeated work.

    Returns:
        ``None``, or an additive mask broadcastable to ``[batch, heads, dim, dim]``.
    """
    codes = parse_mask_spec(spec, num_heads)

    # A soft mask with a zero cap is exactly bidirectional, so rewrite it as one. Doing
    # this here rather than returning None from build_additive_mask is deliberate: the
    # homogeneous branch below dereferences the returned plane with .view() and has no
    # None guard, so a plain "SSSSSSSS" spec would raise AttributeError. Rewriting keeps
    # the invariant that build_additive_mask returns None only for "B", and lets the
    # all-B early return handle the degenerate case for free.
    if soft_cap == 0.0:
        codes = ["B" if code == "S" else code for code in codes]

    if all(code == "B" for code in codes):
        return None

    if len(set(codes)) == 1:
        # Homogeneous: broadcast the shared plane across heads instead of copying it.
        plane = build_additive_mask(codes[0], dim, device, dtype,
                                    soft_cap=soft_cap, soft_tau=soft_tau)
        return plane.view(1, 1, dim, dim)

    key = (tuple(codes), dim, str(device), dtype)
    if "S" in codes:
        key += (float(soft_cap), float(soft_tau))
    cached = _HEAD_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    zeros = None
    planes = []
    for code in codes:
        plane = build_additive_mask(code, dim, device, dtype,
                                    soft_cap=soft_cap, soft_tau=soft_tau)
        if plane is None:  # bidirectional head: contributes nothing
            if zeros is None:
                zeros = torch.zeros(dim, dim, device=device, dtype=dtype)
            plane = zeros
        planes.append(plane)

    stacked = torch.stack(planes, dim=0).unsqueeze(0)  # [1, heads, dim, dim]
    _HEAD_MASK_CACHE[key] = stacked
    return stacked
