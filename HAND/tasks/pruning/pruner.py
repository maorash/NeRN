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


def get_filter_reconstruction_errors(reconstructed_weights, original_weights, filter_type: str):
    filter_reconstruction_errors = list()
    for original_weight, reconstructed_weight in zip(original_weights, reconstructed_weights):
        weight_error = (original_weight - reconstructed_weight).abs()
        weight_error = weight_error / (original_weight.abs())
        if filter_type == '3x3':
            filter_error = weight_error.mean(dim=[2, 3])
        elif filter_type == '3x3xCin':
            filter_error = weight_error.mean(dim=[1, 2, 3])
        else:
            raise ValueError(f"Unsupported filter_type: {filter_type}")
        filter_reconstruction_errors.append(filter_error)
    return filter_reconstruction_errors


def get_prune_indices(tensor_list: List[torch.Tensor], pruning_method: str, pruning_factor: float):
    if pruning_method in ['reconstruction', 'filter_reconstruction']:
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


def get_full_filter_prune_indices(tensor_list: List[torch.Tensor], pruning_factor: float, pruning_method: str):
    prune_indices = list()
    for layer_filter_errors in tensor_list:
        if pruning_method == 'filter_reconstruction':
            all_sorted, all_sorted_idx = torch.sort(layer_filter_errors.view(-1), descending=True)
        elif pruning_method == 'magnitude':
            all_sorted, all_sorted_idx = torch.sort(layer_filter_errors.view(-1), descending=False)
        n = int(layer_filter_errors.size()[0] * pruning_factor)
        prune_indices.append(all_sorted_idx[:n])
    return prune_indices


def prune_weights(original_weights: List[torch.Tensor], indices_to_prune: List[torch.Tensor]):
    with torch.no_grad():
        pruned_weights = copy.deepcopy(original_weights)
        for original_layer, layer_indices_to_prune in zip(pruned_weights, indices_to_prune):
            original_layer.view(-1)[layer_indices_to_prune] = 0
    return pruned_weights


def prune_filters(original_weights: List[torch.Tensor], filter_indices_to_prune: List[torch.Tensor], filter_type: str):
    with torch.no_grad():
        pruned_weights = copy.deepcopy(original_weights)
        for original_layer, layer_indices_to_prune in zip(pruned_weights, filter_indices_to_prune):
            if filter_type == '3x3':
                k = original_layer.size()[2]
                original_layer.view(-1, k, k)[layer_indices_to_prune, :, :] = 0
            elif filter_type == '3x3xCin':
                original_layer[layer_indices_to_prune, :, :, :] = 0
            else:
                raise ValueError(f"Unsupported filter_type: {filter_type}")
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

    def prune(self):
        if self.cfg.pruning_method == 'reconstruction':
            self.reconstruction_prune(self.cfg.pruning_factor, self.cfg.reconstruction_error_metric)
        elif self.cfg.pruning_method == 'magnitude':
            self.magnitude_prune(self.cfg.pruning_factor)
        elif self.cfg.pruning_method == 'random':
            self.random_prune(self.cfg.pruning_factor)

    def reconstruction_prune(self, pruning_factor: float, error_metric: str = 'relative'):
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

    def reconstruction_filter_prune(self, pruning_factor: float, filter_type: str = '3x3'):
        original_weights = self.original_model.get_learnable_weights()
        reconstructed_weights = self.reconstructed_model.get_learnable_weights()

        # calculate reconstruction error
        filter_reconstruction_errors = get_filter_reconstruction_errors(reconstructed_weights, original_weights,
                                                                        filter_type)

        # get indices of weights to prune - those with the largest reconstruction  errors
        if filter_type == '3x3':
            largest_error_filter_indices = get_prune_indices(filter_reconstruction_errors, 'filter_reconstruction',
                                                             pruning_factor)
        elif filter_type == '3x3xCin':
            largest_error_filter_indices = get_full_filter_prune_indices(filter_reconstruction_errors, pruning_factor,
                                                                         'filter_reconstruction')
        else:
            raise ValueError(f"Unsupported filter_type: {filter_type}")

        # prune original weights
        pruned_original_weights = prune_filters(original_weights, largest_error_filter_indices, filter_type)
        self.pruned_model.update_weights(pruned_original_weights)
        return

    def magnitude_filter_prune(self, pruning_factor: float, filter_type: str = '3x3'):
        original_weights = self.original_model.get_learnable_weights()
        filter_magnitudes = list()
        for layer in original_weights:
            weight_magnitudes = layer.abs()
            if filter_type == '3x3':
                filter_magnitude = weight_magnitudes.mean(dim=[2, 3])
            elif filter_type == '3x3xCin':
                filter_magnitude = weight_magnitudes.mean(dim=[1, 2, 3])
            else:
                raise ValueError(f"Unsupported filter_type: {filter_type}")
            filter_magnitudes.append(filter_magnitude)

        # get indices of weights to prune - those with the largest reconstruction  errors
        if filter_type == '3x3':
            smallest_magnitude_filter_indices = get_prune_indices(filter_magnitudes, 'magnitude',
                                                                  pruning_factor)
        elif filter_type == '3x3xCin':
            smallest_magnitude_filter_indices = get_full_filter_prune_indices(filter_magnitudes, pruning_factor,
                                                                              'magnitude')
        else:
            raise ValueError(f"Unsupported filter_type: {filter_type}")

        # prune original weights
        pruned_original_weights = prune_filters(original_weights, smallest_magnitude_filter_indices, filter_type)
        self.pruned_model.update_weights(pruned_original_weights)
        return
