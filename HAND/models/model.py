import copy
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import DataParallel
from typing import List, Tuple
from torch.nn import functional as F

from abc import abstractmethod

from HAND.options import TrainConfig
from HAND.positional_embedding import MyPositionalEncoding
import HAND.permutations.utils as permutations_utils


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
    def __init__(self, original_model: OriginalModel, train_cfg: TrainConfig, sampling_mode: str = "center"):
        super().__init__()
        self.original_model = original_model
        self.reconstructed_model = copy.deepcopy(original_model)
        self.reinitialize_learnable_weights()
        self.embeddings_cfg = train_cfg.hand.embeddings
        self.permutations_cfg = train_cfg.hand.embeddings.permutations_config
        self.original_model_path = train_cfg.original_model_path
        self.positional_encoder = MyPositionalEncoding(train_cfg.hand.embeddings)
        self.sampling_mode = sampling_mode
        self.indices = self._get_tensor_indices()
        self.positional_embeddings = self._calculate_position_embeddings()

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

    def _calculate_unpermuted_position_embeddings(self) -> List[List[torch.Tensor]]:
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

    def _calculate_position_embeddings(self) -> List[torch.Tensor]:
        unpermuted_positional_embeddings = self._calculate_unpermuted_position_embeddings()
        unpermuted_positional_embeddings = [torch.stack(layer_emb).to(self.device) for layer_emb in
                                            unpermuted_positional_embeddings]
        if self.permutations_cfg.permute_mode is None:
            return unpermuted_positional_embeddings

        permutations_cache_folder = Path(
            __file__).parent / f"{str(self)}_permutations_{hash((self.original_model_path, self.permutations_cfg.permute_mode))}"
        os.makedirs(permutations_cache_folder, exist_ok=True)
        try:
            print("Trying to load precomputed permutations")
            permutations = []
            for i in range(len(self.indices)):
                permutations.append(torch.load(permutations_cache_folder / f"layer_{i}.pt"))
                print(f"Loaded permutations for layer {i + 1}/{len(self.indices)}")
            print("Finished loading precomputed permutations")
        except IOError:
            print("Couldn't load precomputed permutations, computing permutations")

            numpy_weights = [layer_weight.detach().cpu().numpy() for layer_weight in
                             self.original_model.get_learnable_weights()]
            if self.permutations_cfg.permute_mode == "joint":
                permutations = permutations_utils.calculate_joint_permutations(numpy_weights,
                                                                               self.permutations_cfg.num_workers)
            elif self.permutations_cfg.permute_mode == "separate":
                permutations = permutations_utils.calculate_separate_permutations(numpy_weights)
            else:
                raise ValueError("Unsupported permutations mode")

            for i in range(len(permutations)):
                print(f"Saving permutations for layer {i + 1}/{len(permutations)}")
                torch.save(permutations[i], permutations_cache_folder / f"layer_{i}.pt")

        if self.permutations_cfg.permute_mode == "joint":
            return permutations_utils.joint_permute(unpermuted_positional_embeddings, permutations)
        elif self.permutations_cfg.permute_mode == "separate":
            return permutations_utils.seperate_permute(unpermuted_positional_embeddings, permutations,
                                                       self.get_learnable_weights_shapes)
        else:
            raise ValueError("Unsupported permutations mode")

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

            curr_layer_weights.data = curr_layer_weights.data * 0. + sampled_predicted_weights

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
