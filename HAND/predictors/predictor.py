import torch
from torch import nn
from typing import List
from abc import ABC, abstractmethod

from HAND.options import HANDConfig
from HAND.predictors.activations import ActivationsFactory


class HANDPredictorFactory:
    def __init__(self, cfg: HANDConfig, input_size: int):
        self.cfg = cfg
        self.input_size = input_size

    def get_predictor(self):
        if self.cfg.method == 'basic':
            return HANDBasicPredictor(self.cfg, self.input_size)
        elif self.cfg.method == '3x3':
            return HAND3x3Predictor(self.cfg, self.input_size)
        else:
            raise ValueError(f'Not recognized predictor type {self.cfg.method}')


class HANDPredictorBase(nn.Module, ABC):
    def __init__(self, cfg: HANDConfig, input_size: int):
        super().__init__()
        self.cfg = cfg
        self.input_size = input_size
        self.act_layer = ActivationsFactory.get(cfg.act_layer)

    @abstractmethod
    def forward(self, positional_embedding: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class HAND3x3Predictor(HANDPredictorBase):
    """
    Given 3 positional embeddings: (Layer, Filter, Channel) returns a 3x3 filter tensor
    """

    def __init__(self, cfg: HANDConfig, input_size: int):
        super().__init__(cfg, input_size)
        self.hidden_size = cfg.hidden_layer_size
        self.layers = self._construct_layers()
        self.final_linear_layer = nn.Linear(self.hidden_size, 9)

    def _construct_layers(self):
        blocks = [nn.Linear(self.input_size, self.hidden_size)]
        blocks.extend([nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.cfg.num_blocks - 2)])
        return nn.ModuleList(blocks)

    def forward(self, positional_embedding: torch.Tensor) -> torch.Tensor:
        x = positional_embedding
        for layer in self.layers:
            x = layer(x)
            x = self.act_layer(x)
        x = self.final_linear_layer(x)
        return x


class HANDBasicPredictor(HANDPredictorBase):
    """
    Given 5 positional embeddings: (Layer, Filter, Channel, Height, Width) returns a single floating point
    """

    def forward(self, positional_embedding: List[torch.Tensor]) -> List[torch.Tensor]:
        pass


