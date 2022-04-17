import torch
from torch import nn
from typing import List
from abc import ABC, abstractmethod


class HANDPredictorBase(nn.Module, ABC):
    @abstractmethod
    def forward(self, positional_embeddings: List[torch.TensorType]) -> List[torch.TensorType]:
        raise NotImplementedError()


class HAND3x3Predictor(HANDPredictorBase):
    """
    Given 3 positional embeddings: (Layer, Filter, Channel) returns a 3x3 filter tensor
    """

    def forward(self, positional_embeddings: List[torch.TensorType]) -> List[torch.TensorType]:
        pass


class HANDBasicPredictor(HANDPredictorBase):
    """
    Given 5 positional embeddings: (Layer, Filter, Channel, Height, Width) returns a single floating point
    """

    def forward(self, positional_embeddings: List[torch.TensorType]) -> List[torch.TensorType]:
        pass
