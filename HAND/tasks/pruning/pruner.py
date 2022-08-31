from typing import List, Union, Tuple

import copy
import torch

from HAND.models.model import OriginalModel, OriginalDataParallel, ReconstructedModel, ReconstructedDataParallel
from HAND.predictors.factory import PredictorDataParallel
from HAND.predictors.predictor import HANDPredictorBase
from HAND.tasks.model_factory import ModelFactory
from HAND.tasks.pruning.prune_options import PruneConfig


def get_reconstruction_errors(reconstructed_weights, original_weights):
    reconstruction_errors = list()
    for original_weight, reconstructed_weight in zip(original_weights, reconstructed_weights):
        error = (original_weight - reconstructed_weight) ** 2
        reconstruction_errors.append(error)
    return reconstruction_errors


# def get_largest_error_indices(reconstruction_errors: List[torch.Tensor], pruning_factor: float):
#     all_sorted, all_sorted_idx = torch.sort(torch.cat([-1 * t.view(-1) for t in reconstruction_errors]))
#     cum_num_elements = torch.cumsum(torch.tensor([t.numel() for t in reconstruction_errors]), dim=0)
#     cum_num_elements = torch.cat([torch.tensor([0]), cum_num_elements])
#
#     n = int(cum_num_elements[-1].item() * pruning_factor)
#     split_indices_lt = [all_sorted_idx[:n] < cum_num_elements[i + 1] for i, _ in enumerate(cum_num_elements[1:])]
#     split_indices_ge = [all_sorted_idx[:n] >= cum_num_elements[i] for i, _ in enumerate(cum_num_elements[:-1])]
#     largest_error_indices = [all_sorted_idx[:n][torch.logical_and(lt, ge)] - c for lt, ge, c in
#                              zip(split_indices_lt, split_indices_ge, cum_num_elements[:-1])]
#
#     # returns list of tensors with linear indices in each tensor
#     return largest_error_indices

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
                 device):
        self.cfg = config
        self.predictor = predictor
        self.original_model = original_model
        self.reconstructed_model = reconstructed_model
        self.device = device

    def prune(self, pruning_factor: float):
        learnable_weights_shapes = self.reconstructed_model.get_learnable_weights_shapes()
        indices, positional_embeddings = self.reconstructed_model.get_indices_and_positional_embeddings()
        positional_embeddings = [torch.stack(layer_emb).to(self.device) for layer_emb in positional_embeddings]

        # get original weights and predict reconstructed weights
        original_weights = self.original_model.get_learnable_weights()
        reconstructed_weights = HANDPredictorBase.predict_all(self.predictor, positional_embeddings,
                                                              original_weights,
                                                              learnable_weights_shapes)
        self.reconstructed_model.update_weights(reconstructed_weights)
        updated_reconstructed_weights = self.reconstructed_model.get_learnable_weights()

        # calculate reconstruction error
        reconstruction_errors = get_reconstruction_errors(updated_reconstructed_weights, original_weights)

        # get indices of weights to prune - those with the largest reconstruction  errors
        largest_error_indices = get_prune_indices(reconstruction_errors, 'reconstruction', pruning_factor)

        # prune original weights
        pruned_original_weights = prune_weights(original_weights, largest_error_indices)

        # create new model with pruned weights
        pruned_model = ModelFactory.models[self.cfg.train_cfg.task.original_model_name][1](self.original_model,
                                                                                           self.cfg.train_cfg.hand.embeddings,
                                                                                           sampling_mode=self.cfg.train_cfg.hand.sampling_mode).to(
            self.device)
        pruned_model.update_weights(pruned_original_weights)

        return pruned_model

    def magnitude_prune(self, pruning_factor: float):
        original_weights = self.original_model.get_learnable_weights()
        smallest_magnitude_weight_indices = get_prune_indices(original_weights, 'magnitude', pruning_factor)
        pruned_original_weights = prune_weights(original_weights, smallest_magnitude_weight_indices)
        pruned_model = ModelFactory.models[self.cfg.train_cfg.task.original_model_name][1](self.original_model,
                                                                                           self.cfg.train_cfg.hand.embeddings,
                                                                                           sampling_mode=self.cfg.train_cfg.hand.sampling_mode).to(
            self.device)
        pruned_model.update_weights(pruned_original_weights)
        return pruned_model
