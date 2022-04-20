from abc import ABC, abstractmethod
from typing import Optional

import torch

from torch import nn

from HAND.models.model import ReconstructedModel, OriginalModel


class LossBase(nn.Module, ABC):
    @abstractmethod
    def forward(self,
                reconstructed_model: ReconstructedModel,
                original_model: OriginalModel,
                batch: Optional[torch.TensorType]) \
            -> torch.Tensor:
        raise NotImplementedError()


class ReconstructionLoss(LossBase):
    def __init__(self, loss_type: str = 'MSELoss'):
        super().__init__()
        self.loss_function = getattr(nn, loss_type)

    def forward(self,
                reconstructed_model: ReconstructedModel,
                original_model: OriginalModel,
                **kwargs) \
            -> torch.Tensor:
        original_weights = original_model.get_learnable_weights()
        reconstructed_weights = reconstructed_model.get_learnable_weights()
        return self.loss_function(original_weights, reconstructed_weights)


class FeatureMapsDistillationLoss(LossBase):
    def __init__(self, loss_type: str = 'MSELoss'):
        super().__init__()
        self.loss_function = getattr(nn, loss_type)

    def forward(self,
                batch: torch.TensorType,
                reconstructed_model: ReconstructedModel,
                original_model: OriginalModel) \
            -> torch.Tensor:
        original_feature_maps = original_model.get_feature_maps(batch)
        reconstructed_feature_maps = reconstructed_model.get_feature_maps(batch)

        loss = torch.Tensor(0.)
        for original_fmap, reconstructed_fmap in zip(original_feature_maps,
                                                     reconstructed_feature_maps):
            loss += self.loss_function(original_fmap / torch.linalg.norm(original_fmap).item(),
                                       reconstructed_fmap / torch.linalg.norm(reconstructed_fmap).item())

        return loss


class OutputDistillationLoss(LossBase):
    def __init__(self, loss_type: str = 'KLDivLoss'):
        super().__init__()
        self.loss_function = getattr(nn, loss_type)

    def forward(self,
                batch: torch.TensorType,
                reconstructed_model: ReconstructedModel,
                original_model: OriginalModel) \
            -> torch.Tensor:
        original_model_output = original_model(batch)
        reconstructed_model_output = reconstructed_model(batch)
        return self.loss_function(original_model_output, reconstructed_model_output)
