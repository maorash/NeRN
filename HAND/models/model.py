import copy

import torch
from torch import nn
from typing import List, Tuple

from abc import abstractmethod


class OriginalModel(nn.Module):
    @abstractmethod
    def get_feature_maps(self, batch: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        pass

    @abstractmethod
    def get_learnable_weights(self) -> List[torch.Tensor]:
        pass

    def get_learnable_weights_norms(self):
        learnable_weights = self.get_learnable_weights()
        norms = [weight.norm() for weight in learnable_weights]
        return norms

    @abstractmethod
    def get_fully_connected_weights(self) -> torch.Tensor:
        pass


class ReconstructedModel(OriginalModel):
    def __init__(self, original_model: OriginalModel):
        super().__init__()
        self.original_model = original_model
        self.reconstructed_model = copy.deepcopy(original_model)
        self.reinitialize_learnable_weights()

    def get_feature_maps(self, batch: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return self.reconstructed_model.get_feature_maps(batch)

    def get_learnable_weights(self) -> List[torch.Tensor]:
        weights = self.reconstructed_model.get_learnable_weights()
        return weights

    def get_learnable_weights_shapes(self) -> List[torch.Size]:
        return [weights.shape for weights in self.reconstructed_model.get_learnable_weights()]

    def get_fully_connected_weights(self) -> torch.Tensor:
        return self.reconstructed_model.get_fully_connected_weights()

    def reinitialize_learnable_weights(self):
        for weight in self.reconstructed_model.get_learnable_weights():
            nn.init.xavier_normal_(weight)

    def _calculate_position_embeddings(self) -> List[List[torch.Tensor]]:
        positional_embeddings = [[self.positional_encoder(idx) for idx in layer_indices] for layer_indices in
                                 self.indices]
        return positional_embeddings

    def get_indices_and_positional_embeddings(self) -> Tuple[List[List[Tuple]], List[List[torch.Tensor]]]:
        return self.indices, self.positional_embeddings

    def update_weights(self, reconstructed_weights: List[torch.Tensor]):
        learnable_weights = self.get_learnable_weights()
        for curr_layer_weights, curr_predicted_weights in zip(learnable_weights, reconstructed_weights):
            curr_layer_weights.data = curr_layer_weights.data * 0. + curr_predicted_weights

    def forward(self, x):
        return self.reconstructed_model(x)


class ReconstructedModelKxK(ReconstructedModel):
    def __init__(self, original_model: OriginalModel):
        super().__init__(original_model)

    def update_weights(self, reconstructed_weights: List[torch.Tensor]):
        learnable_weights = self.get_learnable_weights()
        for curr_layer_weights, curr_predicted_weights in zip(learnable_weights, reconstructed_weights):
            curr_learnable_kernel_size = curr_layer_weights.shape[-1]
            curr_predicted_kernel_size = curr_predicted_weights.shape[-1]
            min_coord = int(curr_predicted_kernel_size / 2 - curr_learnable_kernel_size/2)
            max_coord = int(curr_predicted_kernel_size / 2 + curr_learnable_kernel_size/2)
            curr_layer_weights.data = curr_layer_weights.data * 0. + \
                                      curr_predicted_weights[:, :, min_coord:max_coord, min_coord:max_coord]
