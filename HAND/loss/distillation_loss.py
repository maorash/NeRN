import torch
from torch import nn
from torch.nn import functional as F
from HAND.loss.loss import LossBase


class DistillationLossBase(LossBase):
    def forward(self,
                reconstructed_outputs: torch.Tensor,
                original_outputs: torch.Tensor) \
            -> torch.Tensor:
        raise NotImplementedError()


class KLDivLoss(DistillationLossBase):
    def __init__(self):
        super().__init__()

    def forward(self,
                reconstructed_outputs: torch.Tensor,
                original_outputs: torch.Tensor) \
            -> torch.Tensor:
        return nn.KLDivLoss(reduction="batchmean")(F.log_softmax(reconstructed_outputs, dim=1),
                                                   F.softmax(original_outputs, dim=1))


class StableKLDivLoss(DistillationLossBase):
    def __init__(self):
        super().__init__()

    def forward(self,
                reconstructed_outputs: torch.Tensor,
                original_outputs: torch.Tensor) \
            -> torch.Tensor:
        return nn.KLDivLoss(reduction="batchmean")(torch.log(F.softmax(reconstructed_outputs, dim=1) + 1e-4),
                                                   F.softmax(original_outputs, dim=1))


class DistillationLossFactory:
    losses = {
        "KLDivLoss": KLDivLoss,
        "StableKLDivLoss": StableKLDivLoss
    }

    @staticmethod
    def get(loss_type: str = "KLDivLoss") -> DistillationLossBase:
        try:
            return DistillationLossFactory.losses[loss_type]()
        except KeyError:
            raise ValueError("Unknown Distillation Loss Type")
