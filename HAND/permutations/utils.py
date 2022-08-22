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


def joint_permutations(embeddings: List[torch.Tensor], weights: List[np.array]) -> List[torch.Tensor]:
    embeddings = [embedding.cpu().numpy() for embedding in embeddings]
    permutations = [
        get_max_sim_order(layer_weights.reshape((-1, weights[i].shape[-1] ** 2)),
                          False)
        for i, layer_weights in enumerate(weights)]
    return [embeddings[i][permutations[i]] for i in range(len(embeddings))]


def separate_permutations(embeddings: List[torch.Tensor], weights: List[np.array]) -> List[np.array]:
    reshaped_embeddings = [
        embeddings[i].cpu().numpy().reshape(weights[i].shape[0], weights[i].shape[1], -1) for i in
        range(len(weights))]
    channel_first_weights = [layer_weights.transpose(1, 0, 2, 3) for layer_weights in weights]
    num_layers = len(weights)
    filter_permutations = [
        [get_max_sim_order(filter_weights.reshape((-1, layer_weights.shape[-1] ** 2)), True) for
         filter_weights in layer_weights] for layer_weights in channel_first_weights]
    filter_permuted_weights = [np.array([channel_weights[filter_permutation] for channel_weights, filter_permutation in
                                         zip(channel_first_weights[i], filter_permutations[i])]) for i in
                               range(num_layers)]
    channel_permutations = [
        get_max_sim_order(layer_weights.reshape((-1, layer_weights.shape[1], layer_weights.shape[-1] ** 2)), True)
        for layer_weights in filter_permuted_weights]
    channel_first_embeddings = [reshaped_embeddings[i].transpose(1, 0, 2) for i in range(len(reshaped_embeddings))]

    filter_permuted_embeddings = [
        np.array([channel_embeddings[filter_permutation] for channel_embeddings, filter_permutation in
                  zip(channel_first_embeddings[i], filter_permutations[i])]) for i in
        range(num_layers)]
    permuted_embeddings = [filter_permuted_embeddings[i][channel_permutations[i]].transpose(1, 0, 2) for i in
                           range(num_layers)]

    return [torch.Tensor(permuted_embeddings[i]).reshape(embeddings[i].shape).to(embeddings[i].device) for i in
            range(len(embeddings))]
