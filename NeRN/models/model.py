import copy
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import DataParallel
from typing import List, Tuple
from torch.nn import functional as F

from abc import abstractmethod

from NeRN.log_utils import hash_str
from NeRN.options import Config
from NeRN.positional_embedding import MyPositionalEncoding
from NeRN.permutations.permutations import PermutationsFactory


class OriginalModel(nn.Module):
    @abstractmethod
    def get_feature_maps(self, batch: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        raise NotImplementedError()

    @abstractmethod
    def get_learnable_weights(self) -> List[torch.Tensor]:
        raise NotImplementedError()

    def get_learnable_weights_norms(self):
        learnable_weights = self.get_learnable_weights()
        norms = [weight.norm() for weight in learnable_weights]
        return norms

    @abstractmethod
    def get_fully_connected_weights(self) -> torch.Tensor:
        raise NotImplementedError()

    def load(self, path: str, device: torch.device):
        self.load_state_dict(torch.load(path, map_location=device))


class ReconstructedModel(OriginalModel):
    def __init__(self, original_model: OriginalModel, train_cfg: Config, device: str,
                 sampling_mode: str = "center"):
        super().__init__()
        self.device = device
        self.original_model = original_model
        self.reconstructed_model = copy.deepcopy(original_model)
        self.reinitialize_learnable_weights()
        self.embeddings_cfg = train_cfg.nern.embeddings
        self.permutations_cfg = train_cfg.nern.embeddings.permutations
        self.original_model_path = train_cfg.original_model_path
        self.positional_encoder = MyPositionalEncoding(train_cfg.nern.embeddings)
        self.sampling_mode = sampling_mode
        self.indices = self._get_tensor_indices()
        self.positional_embeddings = self._calculate_positional_embeddings()

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

    def _get_tensor_indices(self) -> List[List[Tuple]]:
        indices = []
        normalize_indices = []

        learnable_weights_shapes = self.get_learnable_weights_shapes()
        max_index = max([max(weights_shape) for weights_shape in learnable_weights_shapes])
        num_layers = len(self.original_model.get_learnable_weights())
        for layer_idx in range(0, num_layers):
            curr_layer_indices = []
            curr_normalized_layer_indices = []
            curr_num_filters = learnable_weights_shapes[layer_idx][0]
            for filter_idx in range(curr_num_filters):
                curr_num_channels = learnable_weights_shapes[layer_idx][1]
                for channel_idx in range(curr_num_channels):
                    curr_layer_indices.append((layer_idx, filter_idx, channel_idx))
                    if self.embeddings_cfg.normalization_mode == "None":
                        curr_normalized_layer_indices.append((layer_idx, filter_idx, channel_idx))
                    elif self.embeddings_cfg.normalization_mode == "global":
                        curr_normalized_layer_indices.append(
                            (layer_idx / max_index, filter_idx / max_index, channel_idx / max_index))
                    elif self.embeddings_cfg.normalization_mode == "local":
                        curr_normalized_layer_indices.append(
                            (layer_idx / num_layers, filter_idx / curr_num_filters, channel_idx / curr_num_channels))
                    else:
                        raise ValueError(f"Unsupported normalization mode {self.embeddings_cfg.normalization_mode}")

            indices.append(curr_layer_indices)
            normalize_indices.append(curr_normalized_layer_indices)

        self.normalized_indices = normalize_indices

        return indices

    def _calculate_unpermuted_positional_embeddings(self) -> List[List[torch.Tensor]]:
        embeddings_cache_folder = Path(__file__).parent / f"{str(self)}_embeddings_{hash(self.positional_encoder)}"
        os.makedirs(embeddings_cache_folder, exist_ok=True)
        try:
            print("Trying to load precomputed embeddings")
            positional_embeddings = []
            for i in range(len(self.indices)):
                positional_embeddings.append(torch.load(embeddings_cache_folder / f"layer_{i}.pt"))
                print(f"Loaded positional embeddings for layer {i + 1}/{len(self.indices)}")
            print("Finished loading precomputed embeddings")
            return positional_embeddings
        except IOError:
            print("Couldn't load precomputed embeddings, computing embeddings")

        positional_embeddings = []
        for i, layer_indices in enumerate(self.indices):
            print(f"Calculating layer {i + 1}/{len(self.indices)} embeddings")
            layer_embeddings = []
            for idx in layer_indices:
                layer_embeddings.append(self.positional_encoder(idx))
            positional_embeddings.append(layer_embeddings)

        for i in range(len(self.indices)):
            print(f"Saving positional embeddings for layer {i + 1}/{len(self.indices)}")
            torch.save(positional_embeddings[i], embeddings_cache_folder / f"layer_{i}.pt")

        return positional_embeddings

    def _calculate_positional_embeddings(self) -> List[torch.Tensor]:
        unpermuted_positional_embeddings = self._calculate_unpermuted_positional_embeddings()
        unpermuted_positional_embeddings = [torch.stack(layer_emb).to(self.device) for layer_emb in
                                            unpermuted_positional_embeddings]
        if self.permutations_cfg.permute_mode is None:
            return unpermuted_positional_embeddings

        permutations_folder_name = f"{str(self)}_permutations_{hash((hash_str(self.original_model_path), hash_str(self.permutations_cfg.permute_mode)))}"
        permutations_cache_folder = Path(__file__).parent / permutations_folder_name
        os.makedirs(permutations_cache_folder, exist_ok=True)

        permuter = PermutationsFactory.get(self.permutations_cfg)
        permutations = []
        loaded = False
        try:
            print("Trying to load precomputed permutations")
            for i in range(len(self.indices)):
                permutations.append(torch.load(permutations_cache_folder / f"layer_{i}.pt"))
                print(f"Loaded permutations for layer {i + 1}/{len(self.indices)}")
            loaded = True
            print("Finished loading precomputed permutations")
        except IOError:
            print("Couldn't load precomputed permutations, computing permutations")

        if not loaded:
            permutations = permuter.calculate(self.original_model.get_learnable_weights())

            for i in range(len(permutations)):
                print(f"Saving permutations for layer {i + 1}/{len(permutations)}")
                torch.save(permutations[i], permutations_cache_folder / f"layer_{i}.pt")

        return permuter.apply(unpermuted_positional_embeddings, permutations, self.get_learnable_weights_shapes())

    def get_indices_and_positional_embeddings(self) -> Tuple[List[List[Tuple]], List[torch.Tensor]]:
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

            curr_layer_weights.data = sampled_predicted_weights

    def forward(self, x):
        return self.reconstructed_model(x)

    @property
    def output_size(self):
        return self.positional_encoder.output_size


class OriginalDataParallel(DataParallel):
    def __init__(self, module: OriginalModel, *args, **kwargs):
        DataParallel.__init__(self, module, *args, **kwargs)
        self.module = module

    def get_learnable_weights(self):
        return self.module.get_learnable_weights()

    def get_feature_maps(self, batch: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return self.module.get_feature_maps(batch)

    def get_fully_connected_weights(self) -> torch.Tensor:
        return self.module.get_fully_connected_weights()

    def load(self, path: str, device: torch.device):
        self.module.load(path, device)


class ReconstructedDataParallel(DataParallel):
    def __init__(self, module: ReconstructedModel, *args, **kwargs):
        # TODO: pass calls to ReconstructedModel in an elegant way, this is a workaround
        DataParallel.__init__(self, module, *args, **kwargs)
        self.module = module

    def get_feature_maps(self, batch: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return self.module.get_feature_maps(batch)

    def get_learnable_weights(self) -> List[torch.Tensor]:
        return self.module.get_learnable_weights()

    def get_learnable_weights_shapes(self) -> List[torch.Size]:
        return self.module.get_learnable_weights_shapes()

    def get_fully_connected_weights(self) -> torch.Tensor:
        return self.module.get_fully_connected_weights()

    def reinitialize_learnable_weights(self):
        self.module.reinitialize_learnable_weights()

    def _calculate_position_embeddings(self) -> List[List[torch.Tensor]]:
        return self.module._calculate_position_embeddings()

    def get_indices_and_positional_embeddings(self) -> Tuple[List[List[Tuple]], List[List[torch.Tensor]]]:
        return self.module.get_indices_and_positional_embeddings()

    def update_weights(self, reconstructed_weights: List[torch.Tensor]):
        self.module.update_weights(reconstructed_weights)

    @property
    def output_size(self):
        return self.module.output_size
