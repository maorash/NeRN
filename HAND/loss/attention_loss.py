import torch

from HAND.loss.loss import LossBase
from HAND.models.model import ReconstructedModel, OriginalModel


class AttentionLossBase(LossBase):
    def forward(self,
                batch: torch.Tensor,
                reconstructed_model: ReconstructedModel,
                original_model: OriginalModel) \
            -> torch.Tensor:
        raise NotImplementedError()


class L2AttentionLoss(AttentionLossBase):
    def __init__(self):
        super().__init__()

    def forward(self,
                batch: torch.Tensor,
                reconstructed_model: ReconstructedModel,
                original_model: OriginalModel) \
            -> torch.Tensor:
        original_feature_maps = original_model.get_feature_maps(batch)
        reconstructed_feature_maps = reconstructed_model.get_feature_maps(batch)

        loss = torch.tensor(0.).to(original_feature_maps[0].device)

        for original_fmap, reconstructed_fmap in zip(original_feature_maps,
                                                     reconstructed_feature_maps):
            normalized_original_fmap = original_fmap / torch.linalg.norm(original_fmap)
            normalized_reconstructed_fmap = reconstructed_fmap / torch.linalg.norm(reconstructed_fmap)
            loss += torch.sum((normalized_original_fmap - normalized_reconstructed_fmap) ** 2)

        return loss


class AttentionLossFactory:
    losses = {
        "L2": L2AttentionLoss
    }

    @staticmethod
    def get(loss_type: str = "L2") -> AttentionLossBase:
        try:
            return AttentionLossFactory.losses[loss_type]()
        except KeyError:
            raise ValueError("Unknown Attention Loss Type")

