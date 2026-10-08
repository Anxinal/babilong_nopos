import torch.nn as nn


class PositionalProbe(nn.Module):
    """MLP from a hidden state to one logit per position (or position bin).

    Small on purpose: with ~1 training example per class, two 1024-wide layers memorised
    the training set (loss 0.001) while test accuracy stayed at 4%. One narrow hidden
    layer with dropout leaves less room for that. ``num_hidden_layers=0`` is a linear
    probe. Returns raw logits: the loss applies the softmax.
    """

    def __init__(self, input_dim, hidden_dim=256, output_dim=2048, num_hidden_layers=1,
                 dropout=0.1):
        super().__init__()
        layers, dim = [], input_dim
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
