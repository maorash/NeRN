import torch
from torch import nn

from HAND.loss.loss import LossBase
from HAND.models.model import ReconstructedModel, OriginalModel


class DistillationLossBase(LossBase):
    def forward(self,
                batch: torch.Tensor,
                reconstructed_model: ReconstructedModel,
                original_model: OriginalModel) \
            -> torch.Tensor:
        raise NotImplementedError()


class KLDistillationLoss(DistillationLossBase):
    def __init__(self):
        super().__init__()

    def forward(self,
                batch: torch.Tensor,
                reconstructed_model: ReconstructedModel,
                original_model: OriginalModel) \
            -> torch.Tensor:
        original_model_output = original_model(batch)
        reconstructed_model_output = reconstructed_model(batch)
        return nn.KLDivLoss(log_target=True, reduction="batchmean")(original_model_output, reconstructed_model_output)


class DistillationLossFactory:
    losses = {
        "KLDivLoss": KLDistillationLoss
    }

    @staticmethod
    def get(loss_type: str = "KLDivLoss") -> DistillationLossBase:
        try:
            return DistillationLossFactory.losses[loss_type]()
        except KeyError:
            raise ValueError("Unknown Distillation Loss Type")
