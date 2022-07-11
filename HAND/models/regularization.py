from abc import ABC, abstractmethod

import torch
from torch import nn

from HAND.models.model import OriginalModel


class RegularizationBase(nn.Module, ABC):
    @abstractmethod
    def forward(self, model: OriginalModel) -> torch.Tensor:
        raise NotImplementedError()


class CosineSmoothness(RegularizationBase):
    def forward(self, model: OriginalModel) -> torch.Tensor:
        learnable_weights = model.get_learnable_weights()
        # total_cosine = 0
        total_cosine = torch.zeros(1).to(learnable_weights[0].device)
        for layer_weights in learnable_weights:
            cout, cin, h, w = layer_weights.shape

            resized_layer_weights = layer_weights.reshape([cout, cin, -1])
            cin_term = torch.zeros_like(total_cosine)
            cout_term = torch.zeros_like(total_cosine)
            if cin > 1:
                shifted = resized_layer_weights[:, 1:, :]
                cin_term = torch.sum(torch.cosine_similarity(resized_layer_weights[:, :-1, :], shifted, dim=2))
            if cout > 1:
                shifted = resized_layer_weights[1:, :, :]
                cout_term = torch.sum(torch.cosine_similarity(resized_layer_weights[:-1, :, :], shifted, dim=2))
            total_cosine += cin_term + cout_term

        return total_cosine


class L2Smoothness(RegularizationBase):
    def forward(self, model: OriginalModel):
        learnable_weights = model.get_learnable_weights()
        # total_l2 = 0
        total_l2 = torch.zeros(1).to(learnable_weights[0].device)
        for layer_weights in learnable_weights:
            cout, cin, h, w = layer_weights.shape

            resized_layer_weights = layer_weights.reshape([cout, cin, -1])
            normalized_layer_weights = resized_layer_weights / torch.norm(resized_layer_weights, dim=2, keepdim=True)
            cin_term = torch.zeros_like(total_l2)
            cout_term = torch.zeros_like(total_l2)
            if cin > 1:
                shifted = resized_layer_weights[:, 1:, :]
                cin_term = torch.sum(torch.norm(normalized_layer_weights[:, :-1, :] - shifted, dim=2))
            if cout > 1:
                shifted = resized_layer_weights[1:, :, :]
                cout_term = torch.sum(torch.norm(normalized_layer_weights[:-1, :, :] - shifted, dim=2))
            total_l2 += cin_term + cout_term

        return -total_l2
