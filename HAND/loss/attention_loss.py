from typing import List

import torch

from HAND.options import HANDConfig
from HAND.loss.loss import LossBase


class AttentionLossBase(LossBase):
    def forward(self,
                reconstructed_feature_maps: List[torch.Tensor],
                original_feature_maps: List[torch.Tensor]) \
            -> torch.Tensor:
        raise NotImplementedError()


class L2AttentionLoss(AttentionLossBase):
    def __init__(self):
        super().__init__()

    def forward(self,
                reconstructed_feature_maps: List[torch.Tensor],
                original_feature_maps: List[torch.Tensor]) \
            -> torch.Tensor:
        loss = torch.tensor(0.).to(original_feature_maps[0].device)

        for original_fmap, reconstructed_fmap in zip(original_feature_maps, reconstructed_feature_maps):
            normalized_original_fmap = original_fmap / torch.linalg.norm(original_fmap)
            normalized_reconstructed_fmap = reconstructed_fmap / torch.linalg.norm(reconstructed_fmap)
            loss += torch.sum((normalized_original_fmap - normalized_reconstructed_fmap) ** 2)

        return loss


class AttentionLossFactory:
    losses = {
        "L2": L2AttentionLoss
    }

    @staticmethod
    def get(cfg: HANDConfig) -> AttentionLossBase:
        try:
            return AttentionLossFactory.losses[cfg.attention_loss_type]()
        except KeyError:
            raise ValueError("Unknown Attention Loss Type")
