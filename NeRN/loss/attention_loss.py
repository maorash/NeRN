from typing import List

import torch
from torch import nn

from NeRN.options import NeRNConfig


class AttentionLossBase(nn.Module):
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
            normalized_reconstructed_fmap = reconstructed_fmap / (torch.linalg.norm(reconstructed_fmap) + 1e-3)
            loss += torch.sum((normalized_original_fmap - normalized_reconstructed_fmap) ** 2)

        return loss


class L2BatchedAttentionLoss(AttentionLossBase):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self,
                reconstructed_feature_maps: List[torch.Tensor],
                original_feature_maps: List[torch.Tensor]) \
            -> torch.Tensor:
        loss = torch.tensor(0.).to(original_feature_maps[0].device)
        batch_size = original_feature_maps[0].size(0)

        for original_fmap, reconstructed_fmap in zip(original_feature_maps, reconstructed_feature_maps):
            loss += torch.sum((self.normalize(original_fmap) - self.normalize(reconstructed_fmap)) ** 2)

        return loss / batch_size

    def normalize(self, m):
        batch_size = m.size(0)
        norm = torch.norm(m.view(batch_size, -1), p=2, dim=1).view(batch_size, 1, 1, 1)
        return m / (norm + self.eps)


class SPAttentionLoss(AttentionLossBase):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self,
                reconstructed_feature_maps: List[torch.Tensor],
                original_feature_maps: List[torch.Tensor]) \
            -> torch.Tensor:
        loss = torch.tensor(0.).to(original_feature_maps[0].device)
        batch_size = original_feature_maps[0].size(0)

        for original_fmap, reconstructed_fmap in zip(original_feature_maps, reconstructed_feature_maps):
            normalized_original_fmap = self.get_g(original_fmap)
            normalized_reconstructed_fmap = self.get_g(reconstructed_fmap)
            loss += (torch.norm(normalized_original_fmap - normalized_reconstructed_fmap, p='fro')) ** 2

        loss *= 1 / (batch_size ** 2)

        return loss

    def get_g(self, mat):
        mat = mat.view(mat.size(0), -1)
        mat_t = torch.transpose(mat, 0, 1)
        g = torch.matmul(mat, mat_t)
        g_norm = torch.norm(g, dim=1).unsqueeze(1)
        return g / (g_norm + self.eps)


class AttentionLossFactory:
    losses = {
        "L2AttentionLoss": L2AttentionLoss,
        "L2BatchedAttentionLoss": L2BatchedAttentionLoss,
        "SPAttentionLoss": SPAttentionLoss
    }

    @staticmethod
    def get(cfg: NeRNConfig) -> AttentionLossBase:
        try:
            return AttentionLossFactory.losses[cfg.attention_loss_type]()
        except KeyError:
            raise ValueError("Unknown Attention Loss Type")
