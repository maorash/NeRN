from abc import ABC, abstractmethod
from typing import Optional, List

import torch

from torch import nn
import torch.nn.functional as F

from HAND.models.model import ReconstructedModel, OriginalModel


class LossBase(nn.Module, ABC):
    @abstractmethod
    def forward(self,
                reconstructed_model: ReconstructedModel,
                original_model: OriginalModel,
                batch: Optional[torch.Tensor]) \
            -> torch.Tensor:
        raise NotImplementedError()


class TaskLoss(LossBase):
    def __init__(self, loss_type: str = 'nll_loss'):
        super().__init__()
        self.loss_function = getattr(F, loss_type)

    def forward(self,
                prediction,
                target,
                **kwargs):
        return self.loss_function(prediction, target)


class ReconstructionLoss(LossBase):
    def __init__(self, loss_type: str = 'MSELoss'):
        super().__init__()
        self.loss_function = getattr(nn, loss_type)()

    def forward(self,
                reconstructed_weights: List[torch.Tensor],
                original_weights: List[torch.Tensor],
                **kwargs) \
            -> torch.Tensor:
        loss = torch.tensor(0.).to(reconstructed_weights[0].device)
        for original_weight, reconstructed_weight in zip(original_weights, reconstructed_weights):
            loss += self.loss_function(original_weight, reconstructed_weight)
        return loss


class FeatureMapsDistillationLoss(LossBase):
    def __init__(self, loss_type: str = 'MSELoss'):
        super().__init__()
        self.loss_function = getattr(nn, loss_type)()

    def forward(self,
                batch: torch.Tensor,
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
        self.loss_function = getattr(nn, loss_type)()

    def forward(self,
                batch: torch.Tensor,
                reconstructed_model: ReconstructedModel,
                original_model: OriginalModel) \
            -> torch.Tensor:
        original_model_output = original_model(batch)
        reconstructed_model_output = reconstructed_model(batch)
        return self.loss_function(original_model_output, reconstructed_model_output)
