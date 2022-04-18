import torch
from torch import nn
from typing import List
from abc import ABC, abstractmethod

from HAND.options import HANDConfig


class HANDPredictorBase(nn.Module, ABC):
    def __init__(self, cfg: HANDConfig):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, positional_embeddings: List[torch.TensorType]) -> List[torch.TensorType]:
        raise NotImplementedError()


class HAND3x3Predictor(HANDPredictorBase):
    """
    Given 3 positional embeddings: (Layer, Filter, Channel) returns a 3x3 filter tensor
    """

    def __init__(self, cfg: HANDConfig):
        super().__init__(cfg)
        self.layers = self._construct_layers()

    def _construct_layers(self):
        input_dim = 21  # TODO: determine from config that also determines PEs (when we agree on an implementation)
        hidden_size = 30
        blocks = [nn.Linear(input_dim, hidden_size)]
        blocks.extend([nn.Linear(hidden_size, hidden_size) for _ in range(self.cfg.num_blocks - 2)])
        blocks.append(nn.Linear(hidden_size, 9))  # last layer predicts 3x3 output
        return nn.ModuleList(blocks)

    def forward(self, positional_embeddings: List[torch.TensorType]) -> List[torch.TensorType]:
        pass


class HANDBasicPredictor(HANDPredictorBase):
    """
    Given 5 positional embeddings: (Layer, Filter, Channel, Height, Width) returns a single floating point
    """

    def forward(self, positional_embeddings: List[torch.TensorType]) -> List[torch.TensorType]:
        pass
