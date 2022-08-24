import json
import os
from typing import List

import pyrallis
import torch
from clearml import Task

from HAND.HAND_train import load_original_model
from HAND.models.model import OriginalDataParallel
from HAND.predictors.factory import HANDPredictorFactory
from HAND.predictors.predictor import HANDPredictorBase
from HAND.tasks.pruning.prune_options import PruneConfig


def get_reconstruction_errors(reconstructed_weights, original_weights):
    reconstruction_errors = list()
    for original_weight, reconstructed_weight in zip(original_weights, reconstructed_weights):
        error = (original_weight - reconstructed_weight) ** 2
        reconstruction_errors.append(error)
    return reconstruction_errors


def get_smallest_error_indices(reconstruction_errors: List[torch.tensor()], pruning_factor: float):
    all_sorted, all_sorted_idx = torch.sort(torch.cat([t.view(-1) for t in reconstruction_errors]))
    cum_num_elements = torch.cumsum(torch.tensor([t.numel() for t in reconstruction_errors]), dim=0)
    cum_num_elements = torch.cat([torch.tensor([0]), cum_num_elements])

    n = cum_num_elements[-1].item() * pruning_factor
    split_indeces_lt = [all_sorted_idx[:n] < cum_num_elements[i + 1] for i, _ in enumerate(cum_num_elements[1:])]
    split_indeces_ge = [all_sorted_idx[:n] >= cum_num_elements[i] for i, _ in enumerate(cum_num_elements[:-1])]
    smallest_error_indices = [all_sorted_idx[:n][torch.logical_and(lt, ge)] - c for lt, ge, c in
                     zip(split_indeces_lt, split_indeces_ge, cum_num_elements[:-1])]

    # n_smallest_errors = [t.view(-1)[idx] for t, idx in zip(reconstruction_errors, smallest_error_indices)]

    # returns list of tensors with linear indices in each tensor
    return smallest_error_indices

@pyrallis.wrap()
def main(cfg: PruneConfig):
    use_cuda = not cfg.train_cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # import original model
    original_model, reconstructed_model = load_original_model(cfg.train_cfg, device)

    # import predictor
    pos_embedding = reconstructed_model.positional_encoder.output_size
    predictor = HANDPredictorFactory(cfg.train_cfg, input_size=pos_embedding).get_predictor().to(device)
    if cfg.is_data_parallel:
        predictor = OriginalDataParallel(predictor)
        predictor.load_state_dict(torch.load(cfg.predictor_path).state_dict())
        predictor = predictor.module
    else:
        predictor.load_state_dict(torch.load(cfg.predictor_path).state_dict())

    learnable_weights_shapes = reconstructed_model.get_learnable_weights_shapes()
    indices, positional_embeddings = reconstructed_model.get_indices_and_positional_embeddings()
    positional_embeddings = [torch.stack(layer_emb).to(device) for layer_emb in positional_embeddings]

    # get original weights and predict reconstructed weights
    original_weights = original_model.get_learnable_weights()
    reconstructed_weights = HANDPredictorBase.predict_all(predictor, positional_embeddings,
                                                          original_weights,
                                                          learnable_weights_shapes)
    reconstructed_model.update_weights(reconstructed_weights)
    updated_reconstructed_weights = reconstructed_model.get_learnable_weights()

    # calculate reconstruction error
    reconstruction_errors = get_reconstruction_errors(updated_reconstructed_weights, original_weights)

    # flat all errors of the network and sort (or get indexes of top precentile numpy)
    # zero weights
    # evaluate pruned model without fine tuning


if __name__ == '__main__':
    main()
