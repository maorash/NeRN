import torch
from torch import nn
import torch.nn.functional as F
from typing import List
from abc import ABC, abstractmethod

from HAND.options import HANDConfig


class HANDPredictorFactory:
    def __init__(self, cfg: HANDConfig):
        self.cfg = cfg

    def get_predictor(self):
        if self.cfg.method == 'basic':
            return HANDBasicPredictor(self.cfg)
        elif self.cfg.method == '3x3':
            return HAND3x3Predictor(self.cfg)
        else:
            raise ValueError(f'Not recognized predictor type {self.cfg.method}')


class HANDPredictorBase(nn.Module, ABC):
    def __init__(self, cfg: HANDConfig):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, positional_embedding: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class HAND3x3Predictor(HANDPredictorBase):
    """
    Given 3 positional embeddings: (Layer, Filter, Channel) returns a 3x3 filter tensor
    """

    def __init__(self, cfg: HANDConfig):
        super().__init__(cfg)
        hidden_size = 30
        self.layers = self._construct_layers(hidden_size)
        self.final_linear_layer = nn.Linear(hidden_size, 9)

    def _construct_layers(self, hidden_size):
        input_dim = 80 * 3 * 2  # TODO: determine from config that also determines PEs (when we agree on an implementation)
        blocks = [nn.Linear(input_dim, hidden_size)]
        blocks.extend([nn.Linear(hidden_size, hidden_size) for _ in range(self.cfg.num_blocks - 2)])
        return nn.ModuleList(blocks)

    def forward(self, positional_embedding: torch.Tensor) -> torch.Tensor:
        x = positional_embedding
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        x = self.final_linear_layer(x)
        return x


class HANDBasicPredictor(HANDPredictorBase):
    """
    Given 5 positional embeddings: (Layer, Filter, Channel, Height, Width) returns a single floating point
    """

    def forward(self, positional_embedding: List[torch.Tensor]) -> List[torch.Tensor]:
        pass
