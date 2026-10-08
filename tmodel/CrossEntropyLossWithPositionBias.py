

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLossWithPositionBias(nn.Module):
    """Cross-entropy over positions, scaled by how far off the predicted position is.

    Each sample's cross-entropy is multiplied by a factor that runs linearly from
    ``penalty_range[1]`` when the prediction sits on the true position to
    ``penalty_range[0]`` when it is as far away as the sequence allows:

        loss = mean( CE_i * (near + (far - near) * distance_i) )

    ``distance_i`` is the EXPECTED distance from the true position under the predicted
    distribution, divided by ``P - 1`` so it lies in ``[0, 1]`` at any sequence length.
    It is an expectation, not the distance of the argmax, so it has a gradient.

    Args:
        penalty_range: ``(far, near)``, the factor at the largest and at zero distance.
        ignore_index:  targets with this value are left out.
    """

    def __init__(self, penalty_range=(1.2, 0.5), ignore_index: int = -100):
        super().__init__()
        self.penalty_far, self.penalty_near = penalty_range
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: ``[N, P]``, one score per position. A position that cannot be the
                answer (e.g. padding) can be given ``-inf``; it then has probability 0
                and adds nothing to the distance.
            target: ``[N]`` true position indices in ``[0, P)``, or ``ignore_index``.

        Returns:
            Scalar: the mean over the targets that are not ignored (``nan`` if all are,
            as ``nn.CrossEntropyLoss`` gives).
        """
        num_positions = logits.size(-1)
        keep = target != self.ignore_index
        logits, target = logits[keep].float(), target[keep]

        ce = F.cross_entropy(logits, target, reduction="none")           # [n]
        probs = F.softmax(logits, dim=-1)
        positions = torch.arange(num_positions, device=logits.device)
        # [n, P]: how far each position is from that row's true position.
        distance = (positions.unsqueeze(0) - target.unsqueeze(1)).abs()
        expected = (probs * distance).sum(dim=-1) / max(num_positions - 1, 1)   # in [0, 1]
        factor = self.penalty_near + (self.penalty_far - self.penalty_near) * expected
        return (ce * factor).mean()
