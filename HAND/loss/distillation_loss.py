import torch
from torch import nn

from HAND.loss.loss import LossBase


class DistillationLossBase(LossBase):
    def forward(self,
                reconstructed_outputs: torch.Tensor,
                original_outputs: torch.Tensor) \
            -> torch.Tensor:
        raise NotImplementedError()


class KLDistillationLoss(DistillationLossBase):
    def __init__(self):
        super().__init__()

    def forward(self,
                reconstructed_outputs: torch.Tensor,
                original_outputs: torch.Tensor) \
            -> torch.Tensor:
        return nn.KLDivLoss(log_target=True, reduction="batchmean")(original_outputs, reconstructed_outputs)


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
