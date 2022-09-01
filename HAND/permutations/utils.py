from typing import List

import numpy as np
import torch

from HAND.permutations.tsp import get_max_sim_order


def permute(embeddings: List[torch.Tensor], weights: List[torch.Tensor], permutations_mode: str = None) -> List[
    torch.Tensor]:
    numpy_weights = [layer_weight.detach().cpu().numpy() for layer_weight in weights]
    if permutations_mode == "joint":
        return joint_permutations(embeddings, numpy_weights)
    elif permutations_mode == "separate":
        return separate_permutations(embeddings, numpy_weights)
    elif permutations_mode is None:
        return embeddings
    else:
        raise ValueError("Unsupported permutations mode")


def calculate_layer_joint_permutation(layer_weights: np.array) -> np.array:
    return get_max_sim_order(layer_weights.reshape((-1, layer_weights.shape[-1] ** 2)), False)


def joint_permutations(embeddings: List[torch.Tensor], weights: List[np.array]) -> List[torch.Tensor]:
    from multiprocessing import Pool
    with Pool(5) as p:
        permutations = p.map(calculate_layer_joint_permutation, weights)

    # Do not remove this - currently used for debugging
    permuted_weights = [
        weights[i].reshape((-1, weights[i].shape[-1], weights[i].shape[-1]))[permutations[i]].reshape(weights[i].shape)
        for i in range(len(weights))]
    permuted_weights = [torch.Tensor(permuted_weights[i]).reshape(permuted_weights[i].shape) for i in
                        range(len(permuted_weights))]
    return [embeddings[i][np.argsort(permutations[i])] for i in range(len(embeddings))]


def separate_permutations(embeddings: List[torch.Tensor], weights: List[np.array]) -> List[np.array]:
    reshaped_embeddings = [embeddings[i].cpu().numpy().reshape(weights[i].shape[0], weights[i].shape[1], -1) for i in
                           range(len(weights))]

    num_layers = len(weights)
    in_filter_permutations = [[get_max_sim_order(filter_weights.reshape((-1, layer_weights.shape[-1] ** 2)), True) for
                               filter_weights in layer_weights] for layer_weights in weights]
    in_filter_permuted_weights = [np.array([filter_weights[filter_permutation] for filter_weights, filter_permutation
                                            in zip(weights[i], in_filter_permutations[i])]) for i in range(num_layers)]

    filter_permutations = [
        get_max_sim_order(layer_weights.reshape((-1, layer_weights.shape[1], layer_weights.shape[-1] ** 2)), True)
        for layer_weights in in_filter_permuted_weights]

    # Step 1: apply filter permutations (when handling the embeddings we must apply permutations in reverse)
    # The argsort is necessary because we need the inverse permutation to get the original order
    filter_permuted_embeddings = [reshaped_embeddings[i][np.argsort(filter_permutations[i])] for i in range(num_layers)]

    # Step 2: apply channel (in-filter) permutations
    in_filter_permuted_embeddings = [
        np.array([filter_embeddings[inverse_permutation] for filter_embeddings, inverse_permutation in
                  zip(filter_permuted_embeddings[i], np.argsort(in_filter_permutations[i]))]) for i in
        range(num_layers)]
    # Do not remove this - currently used for debugging
    permuted_weights = [in_filter_permuted_weights[i][filter_permutations[i]] for i in range(num_layers)]

    return [torch.Tensor(in_filter_permuted_embeddings[i]).reshape(embeddings[i].shape).to(embeddings[i].device)
            for i in range(len(embeddings))]
