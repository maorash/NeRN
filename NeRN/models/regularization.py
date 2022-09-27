from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn

from NeRN.models.model import OriginalModel


class RegularizationBase(nn.Module, ABC):
    @abstractmethod
    def forward(self, model: OriginalModel) -> torch.Tensor:
        raise NotImplementedError()


class CosineSmoothness(RegularizationBase):
    def forward(self, model: OriginalModel) -> Tuple[torch.Tensor, torch.Tensor]:
        learnable_weights = model.get_learnable_weights()
        total_cosine = torch.zeros(1).to(learnable_weights[0].device)
        total_l2 = torch.zeros(1).to(learnable_weights[0].device)
        for layer_weights in learnable_weights:
            curr_cosine, curr_l2 = CosineSmoothness.cosine_layer_smoothness(layer_weights)
            total_cosine += curr_cosine
            total_l2 += curr_l2
        return total_cosine, total_l2

    @staticmethod
    def cosine_layer_smoothness(layer_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cout, cin, h, w = layer_weights.shape
        resized_layer_weights = layer_weights.reshape([cout, cin, -1])
        if h == 1 and w == 1:
            return torch.zeros(1).to(resized_layer_weights.device), -L2Smoothness.l2_layer_smoothness(layer_weights)

        cin_term = torch.zeros(1, dtype=layer_weights.dtype, device=layer_weights.device)
        cout_term = torch.zeros(1, dtype=layer_weights.dtype, device=layer_weights.device)
        if cin > 1:
            shifted = resized_layer_weights[:, 1:, :]
            cin_term = torch.sum(torch.cosine_similarity(resized_layer_weights[:, :-1, :], shifted, dim=2))
        if cout > 1:
            shifted = resized_layer_weights[1:, :, :]
            cout_term = torch.sum(torch.cosine_similarity(resized_layer_weights[:-1, :, :], shifted, dim=2))
        return cin_term + cout_term, torch.zeros(1).to(resized_layer_weights.device)


class L2Smoothness(RegularizationBase):
    def forward(self, model: OriginalModel):
        learnable_weights = model.get_learnable_weights()
        total_l2 = torch.zeros(1).to(learnable_weights[0].device)
        for layer_weights in learnable_weights:
            total_l2 += L2Smoothness.l2_layer_smoothness(layer_weights)
        return -total_l2

    @staticmethod
    def l2_layer_smoothness(layer_weights: torch.Tensor) -> torch.Tensor:
        cout, cin, h, w = layer_weights.shape

        resized_layer_weights = layer_weights.reshape([cout, cin, -1])
        if h > 1 or w > 1:
            normalized_layer_weights = resized_layer_weights / torch.norm(resized_layer_weights, dim=2, keepdim=True)
        else:
            normalized_layer_weights = resized_layer_weights / torch.norm(resized_layer_weights, keepdim=True)

        cin_term = torch.zeros(1, dtype=layer_weights.dtype, device=layer_weights.device)
        cout_term = torch.zeros(1, dtype=layer_weights.dtype, device=layer_weights.device)

        if cin > 1:
            shifted = resized_layer_weights[:, 1:, :]
            cin_term = torch.sum(torch.norm(normalized_layer_weights[:, :-1, :] - shifted, dim=2))
        if cout > 1:
            shifted = resized_layer_weights[1:, :, :]
            cout_term = torch.sum(torch.norm(normalized_layer_weights[:-1, :, :] - shifted, dim=2))

        return cin_term + cout_term
