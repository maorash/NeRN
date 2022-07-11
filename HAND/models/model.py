import copy

import torch
from torch import nn
from typing import List, Tuple

from abc import abstractmethod


class OriginalModel(nn.Module):
    @abstractmethod
    def get_feature_maps(self, batch: torch.Tensor) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def get_learnable_weights(self) -> List[torch.Tensor]:
        pass


class ReconstructedModel(OriginalModel):
    def __init__(self, original_model: OriginalModel):
        super().__init__()
        self.original_model = original_model
        self.reconstructed_model = copy.deepcopy(original_model)
        self.reinitialize_learnable_weights()

    @abstractmethod
    def get_indices_and_positional_embeddings(self) -> Tuple[List[Tuple], List[torch.Tensor]]:
        pass

    @abstractmethod
    def aggregate_predicted_weights(self, predicted_weights_raw: List[torch.Tensor]) -> List[torch.Tensor]:
        pass

    def get_feature_maps(self, batch: torch.Tensor) -> List[torch.Tensor]:
        return self.reconstructed_model.get_feature_maps(batch)

    def get_learnable_weights(self) -> List[torch.Tensor]:
        weights = self.reconstructed_model.get_learnable_weights()
        return weights

    def reinitialize_learnable_weights(self):
        for weight in self.reconstructed_model.get_learnable_weights():
            nn.init.xavier_normal_(weight)

    def update_whole_weights(self, aggregated_weights: List[torch.Tensor]):
        for i, aggregated_weight in enumerate(aggregated_weights):
            self.reconstructed_model.get_learnable_weights()[i].data = \
                self.reconstructed_model.get_learnable_weights()[i] * 0. + aggregated_weight

    def forward(self, x):
        return self.reconstructed_model(x)
