import torch
from torch import nn
from torch.nn import functional as F

from NeRN.options import NeRNConfig


class DistillationLossBase(nn.Module):
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


class TempStableKLDivLoss(DistillationLossBase):
    def __init__(self, temperature: float = 2):
        super().__init__()
        self.temperature = temperature

    def forward(self,
                reconstructed_outputs: torch.Tensor,
                original_outputs: torch.Tensor) \
            -> torch.Tensor:
        reconstructed_outputs = reconstructed_outputs / self.temperature
        original_outputs = original_outputs / self.temperature
        return nn.KLDivLoss(reduction="batchmean")(torch.log(F.softmax(reconstructed_outputs, dim=1) + 1e-4),
                                                   F.softmax(original_outputs, dim=1)) * (self.temperature ** 2)


class DistillationLossFactory:
    losses = {
        "KLDivLoss": KLDivLoss,
        "StableKLDivLoss": StableKLDivLoss,
        "TempStableKLDivLoss": TempStableKLDivLoss,
    }

    @staticmethod
    def get(cfg: NeRNConfig) -> DistillationLossBase:
        try:
            return DistillationLossFactory.losses[cfg.distillation_loss_type]()
        except KeyError:
            raise ValueError("Unknown Distillation Loss Type")
