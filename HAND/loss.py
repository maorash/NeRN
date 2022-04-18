from abc import ABC, abstractmethod
from typing import Tuple

import torch

from torch import nn

from HAND.model import ReconstructedModel, OriginalModel


class LossBase(nn.Module, ABC):
    @abstractmethod
    def forward(self,
                batch: torch.TensorType,
                reconstructed_model: ReconstructedModel,
                original_model: OriginalModel) \
            -> torch.FloatType:
        raise NotImplementedError()


class ReconstructionLoss(LossBase):
    # TODO: should `loss_type` be configurable or different implementation for each loss?
    def __init__(self, loss_type: str = 'MSELoss'):
        super().__init__()
        self.loss_function = getattr(nn, loss_type)

    def forward(self,
                batch: torch.TensorType,
                reconstructed_model: ReconstructedModel,
                original_model: OriginalModel) \
            -> torch.FloatType:
        original_model_feature_maps = original_model.get_feature_maps(batch)
        reconstructed_model_feature_maps = reconstructed_model.get_feature_maps(batch)
        return self.loss_function(original_model_feature_maps, reconstructed_model_feature_maps)


class DistillationLoss(LossBase):
    # TODO: should `loss_type` be configurable or different implementation for each loss?
    def __init__(self, loss_type: str = 'CrossEntropyLoss'):
        super().__init__()
        self.loss_function = getattr(nn, loss_type)

    def forward(self,
                batch: torch.TensorType,
                reconstructed_model: ReconstructedModel,
                original_model: OriginalModel) \
            -> torch.FloatType:
        original_model_output = original_model(batch)
        reconstructed_model_output = reconstructed_model(batch)
        return self.loss_function(original_model_output, reconstructed_model_output)
