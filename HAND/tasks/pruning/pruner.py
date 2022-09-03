from typing import List, Union, Tuple

import copy
import torch
import numpy as np

from HAND.models.model import OriginalModel, OriginalDataParallel, ReconstructedModel, ReconstructedDataParallel
from HAND.predictors.factory import PredictorDataParallel
from HAND.predictors.predictor import HANDPredictorBase
from HAND.tasks.pruning.prune_options import PruneConfig


def get_absolute_reconstruction_errors(reconstructed_weights, original_weights):
    reconstruction_errors = list()
    for original_weight, reconstructed_weight in zip(original_weights, reconstructed_weights):
        error = (original_weight - reconstructed_weight) ** 2
        reconstruction_errors.append(error)
    return reconstruction_errors


def get_relative_reconstruction_errors(reconstructed_weights, original_weights):
    reconstruction_errors = list()
    for original_weight, reconstructed_weight in zip(original_weights, reconstructed_weights):
        error = (original_weight - reconstructed_weight).abs()
        error = error / (original_weight.abs())
        reconstruction_errors.append(error)
    return reconstruction_errors


def get_relative_squared_reconstruction_errors(reconstructed_weights, original_weights):
    reconstruction_errors = list()
    for original_weight, reconstructed_weight in zip(original_weights, reconstructed_weights):
        error = (original_weight - reconstructed_weight) ** 2
        error = error / (original_weight.abs())
        reconstruction_errors.append(error)
    return reconstruction_errors


def get_prune_indices(tensor_list: List[torch.Tensor], pruning_method: str, pruning_factor: float):
    if pruning_method == 'reconstruction':
        # the function finds n_smallest so for *largest* errors multiply by -1
        all_sorted, all_sorted_idx = torch.sort(torch.cat([-1 * t.view(-1) for t in tensor_list]))
    elif pruning_method == 'magnitude':
        # the function finds n_smallest so for smallest magnitudes use abs values of the weights
        all_sorted, all_sorted_idx = torch.sort(torch.cat([torch.abs(t.view(-1)) for t in tensor_list]))
    else:
        raise ValueError(f"Unsupported pruning method: {pruning_method}")

    cum_num_elements = torch.cumsum(torch.tensor([t.numel() for t in tensor_list]), dim=0)
    cum_num_elements = torch.cat([torch.tensor([0]), cum_num_elements])

    n = int(cum_num_elements[-1].item() * pruning_factor)
    split_indices_lt = [all_sorted_idx[:n] < cum_num_elements[i + 1] for i, _ in enumerate(cum_num_elements[1:])]
    split_indices_ge = [all_sorted_idx[:n] >= cum_num_elements[i] for i, _ in enumerate(cum_num_elements[:-1])]
    prune_indices = [all_sorted_idx[:n][torch.logical_and(lt, ge)] - c for lt, ge, c in
                             zip(split_indices_lt, split_indices_ge, cum_num_elements[:-1])]

    # returns list of tensors with linear indices in each tensor
    return prune_indices


def prune_weights(original_weights: List[torch.Tensor], indices_to_prune: List[torch.Tensor]):
    with torch.no_grad():
        pruned_weights = copy.deepcopy(original_weights)
        for original_layer, layer_indices_to_prune in zip(pruned_weights, indices_to_prune):
            original_layer.view(-1)[layer_indices_to_prune] = 0
    return pruned_weights


class Pruner:
    def __init__(self, config: PruneConfig,
                 predictor: Union[HANDPredictorBase, PredictorDataParallel],
                 original_model: Union[OriginalModel, OriginalDataParallel],
                 reconstructed_model: Union[ReconstructedModel, ReconstructedDataParallel],
                 pruned_model: Union[ReconstructedModel, ReconstructedDataParallel],
                 device):
        self.cfg = config
        self.predictor = predictor
        self.original_model = original_model
        self.reconstructed_model = reconstructed_model
        self.pruned_model = pruned_model
        self.device = device

    def prune(self, pruning_factor: float, error_metric: bool = False):
        original_weights = self.original_model.get_learnable_weights()
        reconstructed_weights = self.reconstructed_model.get_learnable_weights()
        # calculate reconstruction error
        if error_metric == 'absolute':
            reconstruction_errors = get_absolute_reconstruction_errors(reconstructed_weights, original_weights)
        elif error_metric == 'relative':
            reconstruction_errors = get_relative_reconstruction_errors(reconstructed_weights, original_weights)
        elif error_metric == "relative_squared":
            reconstruction_errors = get_relative_squared_reconstruction_errors(reconstructed_weights, original_weights)
        else:
            raise ValueError(f"Unsupported error_metric: {error_metric}")

        # get indices of weights to prune - those with the largest reconstruction  errors
        largest_error_indices = get_prune_indices(reconstruction_errors, 'reconstruction', pruning_factor)

        # prune original weights
        pruned_original_weights = prune_weights(original_weights, largest_error_indices)
        self.pruned_model.update_weights(pruned_original_weights)
        return

    def magnitude_prune(self, pruning_factor: float):
        original_weights = self.original_model.get_learnable_weights()
        smallest_magnitude_weight_indices = get_prune_indices(original_weights, 'magnitude', pruning_factor)
        pruned_original_weights = prune_weights(original_weights, smallest_magnitude_weight_indices)
        self.pruned_model.update_weights(pruned_original_weights)
        return

    def random_prune(self, pruning_factor: float):
        original_weights = self.original_model.get_learnable_weights()
        previous_shapes = [w.shape for w in original_weights]
        pruned_original_weights = copy.deepcopy(original_weights)
        flattened_original_weights = [w.reshape(-1) for w in pruned_original_weights]
        flattened_original_sizes = [w.shape for w in flattened_original_weights]
        concat_flattened_pruned_weights = torch.cat(flattened_original_weights)
        indices_to_prune = torch.randperm(concat_flattened_pruned_weights.shape[0])[
                           :int(concat_flattened_pruned_weights.shape[0] * pruning_factor)]
        concat_flattened_pruned_weights[indices_to_prune] = 0
        split_indices = np.cumsum([0] + [s[0] for s in flattened_original_sizes])
        split_flattened_pruned_weights = [concat_flattened_pruned_weights[split_indices[i]:split_indices[i + 1]] for i
                                          in range(len(split_indices) - 1)]
        pruned_weights = [w.reshape(s) for w, s in zip(split_flattened_pruned_weights, previous_shapes)]
        self.pruned_model.update_weights(pruned_weights)
        return
