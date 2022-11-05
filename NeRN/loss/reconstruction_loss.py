from typing import List

import torch
from torch import nn

from NeRN.options import NeRNConfig


class ReconstructionLossBase(nn.Module):
    def forward(self,
                reconstructed_weights: List[torch.Tensor],
                original_weights: List[torch.Tensor],
                **kwargs) \
            -> torch.Tensor:
        raise NotImplementedError()


class MSEReconstructionLoss(ReconstructionLossBase):
    def __init__(self):
        super().__init__()

    def forward(self,
                reconstructed_weights: List[torch.Tensor],
                original_weights: List[torch.Tensor],
                **kwargs) \
            -> torch.Tensor:
        mse_loss = nn.MSELoss()
        loss = torch.tensor(0.).to(reconstructed_weights[0].device)
        for original_weight, reconstructed_weight in zip(original_weights, reconstructed_weights):
            loss += mse_loss(original_weight, reconstructed_weight)
        loss /= len(original_weights)
        return loss


class L2ReconstructionLoss(ReconstructionLossBase):
    def __init__(self):
        super().__init__()

    def forward(self,
                reconstructed_weights: List[torch.Tensor],
                original_weights: List[torch.Tensor],
                **kwargs) \
            -> torch.Tensor:
        loss = torch.tensor(0.).to(reconstructed_weights[0].device)
        num_parameters = 0
        for original_weight, reconstructed_weight in zip(original_weights, reconstructed_weights):
            num_parameters += original_weight.numel()
            loss += torch.sum((original_weight - reconstructed_weight) ** 2)
        loss /= num_parameters
        return torch.sqrt(loss)


class ReconstructionLossFactory:
    losses = {
        "MSELoss": MSEReconstructionLoss,
        "L2Loss": L2ReconstructionLoss
    }

    @staticmethod
    def get(cfg: NeRNConfig) -> ReconstructionLossBase:
        try:
            return ReconstructionLossFactory.losses[cfg.reconstruction_loss_type]()
        except KeyError:
            raise ValueError("Unknown Reconstruction Loss Type")
