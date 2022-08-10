import pickle
import copy

import torch
from torch import nn
from typing import List, Tuple
from torch.nn import functional as F

from abc import abstractmethod

from HAND.options import EmbeddingsConfig
from HAND.positional_embedding import MyPositionalEncoding


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
    def __init__(self, original_model: OriginalModel, embeddings_cfg: EmbeddingsConfig, sampling_mode: str = "center"):
        super().__init__()
        self.original_model = original_model
        self.reconstructed_model = copy.deepcopy(original_model)
        self.reinitialize_learnable_weights()
        self.embeddings_cfg = embeddings_cfg
        self.positional_encoder = MyPositionalEncoding(embeddings_cfg)
        self.sampling_mode = sampling_mode

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
        embeddings_cache_filename = f"{str(self)}_embeddings_{hash(self.positional_encoder)}.pkl"
        try:
            print("Trying to load precomputed embeddings")
            with open(embeddings_cache_filename, "rb") as f:
                positional_embeddings = pickle.load(f)
            print("Loaded precomputed embeddings")
            return positional_embeddings
        except IOError:
            print("Couldn't load precomputed embeddings, computing embeddings")

        positional_embeddings = []
        for i, layer_indices in enumerate(self.indices):
            print(f"Calculating layer {i}/{len(self.indices)} embeddings")
            layer_embeddings = []
            for idx in layer_indices:
                layer_embeddings.append(self.positional_encoder(idx))
            positional_embeddings.append(layer_embeddings)

        with open(embeddings_cache_filename, "wb") as f:
            pickle.dump(positional_embeddings, f)
            print("Saved computed embeddings")

        return positional_embeddings

    def get_indices_and_positional_embeddings(self) -> Tuple[List[List[Tuple]], List[List[torch.Tensor]]]:
        return self.indices, self.positional_embeddings

    def update_weights(self, reconstructed_weights: List[torch.Tensor]):
        learnable_weights = self.get_learnable_weights()
        for curr_layer_weights, curr_predicted_weights in zip(learnable_weights, reconstructed_weights):
            # Assuming a square filter
            curr_learnable_kernel_size = curr_layer_weights.shape[-1]
            curr_predicted_kernel_size = curr_predicted_weights.shape[-1]
            if self.sampling_mode == "center":
                min_coord = int(curr_predicted_kernel_size / 2 - curr_learnable_kernel_size / 2)
                max_coord = int(curr_predicted_kernel_size / 2 + curr_learnable_kernel_size / 2)
                sampled_predicted_weights = curr_predicted_weights[:, :, min_coord:max_coord, min_coord:max_coord]
            elif self.sampling_mode == "average":
                sampled_predicted_weights = F.avg_pool2d(curr_predicted_weights,
                                                         curr_predicted_kernel_size - curr_learnable_kernel_size + 1, 1)
            elif self.sampling_mode == "max":
                sampled_predicted_weights = F.max_pool2d(curr_predicted_weights,
                                                         curr_predicted_kernel_size - curr_learnable_kernel_size + 1, 1)
            else:
                raise ValueError(f"Unsupported sampling mode {self.sampling_mode}")

            curr_layer_weights.data = curr_layer_weights.data * 0. + sampled_predicted_weights

    def forward(self, x):
        return self.reconstructed_model(x)
