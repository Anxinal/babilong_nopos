from dataclasses import dataclass


@dataclass
class ALiBiConfig:
    num_layers: int = 6  # in the encoder and in the decoder
    d_model: int = 512
    num_heads: int = 8
    dropout: float = 0.1
