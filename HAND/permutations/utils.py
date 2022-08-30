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
    permutations = [
        get_max_sim_order(layer_weights.reshape((-1, weights[i].shape[-1] ** 2)),
                          False)
        for i, layer_weights in enumerate(weights)]
    # Do not remove this - currently used for debugging
    permuted_weights = [
        weights[i].reshape((-1, weights[i].shape[-1], weights[i].shape[-1]))[permutations[i]].reshape(weights[i].shape)
        for i in range(len(weights))]
    permuted_weights = [torch.Tensor(permuted_weights[i]).reshape(permuted_weights[i].shape) for i in
                        range(len(permuted_weights))]
    return [embeddings[i][np.argsort(permutations[i])] for i in range(len(embeddings))]


def separate_permutations(embeddings: List[torch.Tensor], weights: List[np.array]) -> List[np.array]:
    reshaped_embeddings = [
        embeddings[i].cpu().numpy().reshape(weights[i].shape[0], weights[i].shape[1], -1) for i in
        range(len(weights))]
    cin_first_weights = [layer_weights.transpose(1, 0, 2, 3) for layer_weights in weights]
    num_layers = len(weights)
    filter_permutations = [
        [get_max_sim_order(filter_weights.reshape((-1, layer_weights.shape[-1] ** 2)), True) for
         filter_weights in layer_weights] for layer_weights in cin_first_weights]
    filter_permuted_weights = [np.array([channel_weights[filter_permutation] for channel_weights, filter_permutation in
                                         zip(cin_first_weights[i], filter_permutations[i])]) for i in
                               range(num_layers)]
    channel_permutations = [
        get_max_sim_order(layer_weights.reshape((-1, layer_weights.shape[1], layer_weights.shape[-1] ** 2)), True)
        for layer_weights in filter_permuted_weights]
    cin_first_embeddings = [reshaped_embeddings[i].transpose(1, 0, 2) for i in range(len(reshaped_embeddings))]

    # The argsort is necessary because we need the inverse permutation to get the original order
    filter_permuted_embeddings = [
        np.array([channel_embeddings[inverse_permutations] for channel_embeddings, inverse_permutations in
                  zip(cin_first_embeddings[i], np.argsort(filter_permutations[i]))]) for i in
        range(num_layers)]
    permuted_embeddings = [filter_permuted_embeddings[i][np.argsort(channel_permutations[i])].transpose(1, 0, 2) for i in
                           range(num_layers)]
    # Do not remove this - currently used for debugging
    permuted_weights = [filter_permuted_weights[i][channel_permutations[i]] for i in range(num_layers)]

    return [torch.Tensor(permuted_embeddings[i]).reshape(embeddings[i].shape).to(embeddings[i].device) for i in range(len(embeddings))]


def separate_permutations_two_opt(embeddings: List[torch.Tensor], weights: List[np.array]) -> List[np.array]:
    from HAND.permutations.tsp import two_opt
    from tqdm import tqdm
    reshaped_embeddings = [
        embeddings[i].cpu().numpy().reshape(weights[i].shape[0], weights[i].shape[1], -1) for i in
        range(len(weights))]
    cin_first_weights = [layer_weights.transpose(1, 0, 2, 3) for layer_weights in weights]
    num_layers = len(weights)
    filter_permutations = [
        [two_opt(filter_weights.reshape((-1, layer_weights.shape[-1] ** 2)), 0.01) for
         filter_weights in layer_weights] for layer_weights in tqdm(cin_first_weights)]
    filter_permuted_weights = [np.array([channel_weights[filter_permutation] for channel_weights, filter_permutation in
                                         zip(cin_first_weights[i], filter_permutations[i])]) for i in
                               range(num_layers)]
    channel_permutations = [
        two_opt(layer_weights.reshape((-1, layer_weights.shape[1], layer_weights.shape[-1] ** 2)), 0.01)
        for layer_weights in tqdm(filter_permuted_weights)]
    cin_first_embeddings = [reshaped_embeddings[i].transpose(1, 0, 2) for i in range(len(reshaped_embeddings))]

    # The argsort is necessary because we need the inverse permutation to get the original order
    filter_permuted_embeddings = [
        np.array([channel_embeddings[inverse_permutations] for channel_embeddings, inverse_permutations in
                  zip(cin_first_embeddings[i], np.argsort(filter_permutations[i]))]) for i in
        range(num_layers)]
    permuted_embeddings = [filter_permuted_embeddings[i][np.argsort(channel_permutations[i])].transpose(1, 0, 2) for i in
                           range(num_layers)]
    # Do not remove this - currently used for debugging
    permuted_weights = [filter_permuted_weights[i][np.argsort(channel_permutations[i])] for i in
                        range(num_layers)]

    return [torch.Tensor(permuted_embeddings[i].transpose(1, 0, 2)).reshape(embeddings[i].shape).to(embeddings[i].device) for i in range(len(embeddings))]
