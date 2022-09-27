from typing import List, Tuple

import numpy as np
import torch

from NeRN.permutations.tsp import get_max_sim_order
from NeRN.options import PermutationsConfig


class PermutationsFactory:
    @staticmethod
    def get(permutations_cfg: PermutationsConfig):
        if permutations_cfg.permute_mode == "crossfilter":
            return CrossFilterPermutations(permutations_cfg.num_workers)
        elif permutations_cfg.permute_mode == "infilter":
            return InFilterPermutations()
        else:
            raise ValueError("Unsupported permutations mode")


class CrossFilterPermutations:
    def __init__(self, num_workers: int = 1):
        self.num_workers = num_workers

    def _calculate_layer_crossfilter_permutation(self, layer_weights: np.array) -> np.array:
        return get_max_sim_order(layer_weights.reshape((-1, layer_weights.shape[-1] ** 2)), False)

    def calculate(self, weights: List[np.array], *args, **kwargs) -> List[np.array]:
        numpy_weights = [layer_weight.detach().cpu().numpy() for layer_weight in weights]
        from multiprocessing import Pool
        with Pool(self.num_workers) as p:
            permutations = p.map(self._calculate_layer_crossfilter_permutation, numpy_weights)

        return permutations

    def apply(self, embeddings: List[torch.Tensor], permutations: List[np.array], *args, **kwargs) -> List[
        torch.Tensor]:
        return [embeddings[i][np.argsort(permutations[i])] for i in range(len(embeddings))]


class InFilterPermutations:
    def calculate(self, weights: List[np.array], *args, **kwargs) -> List[Tuple[List[int], List[List[int]]]]:
        numpy_weights = [layer_weight.detach().cpu().numpy() for layer_weight in weights]
        num_layers = len(numpy_weights)
        in_filter_permutations = [
            [get_max_sim_order(filter_weights.reshape((-1, layer_weights.shape[-1] ** 2)), True) for
             filter_weights in layer_weights] for layer_weights in numpy_weights]
        in_filter_permuted_weights = [
            np.array([filter_weights[filter_permutation] for filter_weights, filter_permutation
                      in zip(numpy_weights[i], in_filter_permutations[i])]) for i in range(num_layers)]

        filter_permutations = [
            get_max_sim_order(layer_weights.reshape((-1, layer_weights.shape[1], layer_weights.shape[-1] ** 2)), True)
            for layer_weights in in_filter_permuted_weights]

        return [(layer_filter_permutations, layer_in_filter_permutations) for
                layer_filter_permutations, layer_in_filter_permutations in
                zip(filter_permutations, in_filter_permutations)]

    def apply(self, embeddings: List[torch.Tensor], permutations: List[Tuple[List[int], List[List[int]]]],
              weights_shapes: List[torch.Size], *args, **kwargs) -> List[torch.Tensor]:
        # Each layer's permutation is a tuple of (filter permutations, channel (in-filter) permutations)
        reshaped_embeddings = [embeddings[i].cpu().numpy().reshape(weights_shapes[i][0], weights_shapes[i][1], -1) for
                               i in
                               range(len(weights_shapes))]
        num_layers = len(weights_shapes)
        # Step 1: apply filter permutations (when handling the embeddings we must apply permutations in reverse)
        # The argsort is necessary because we need the inverse permutation to get the original order
        filter_permuted_embeddings = [reshaped_embeddings[i][np.argsort(permutations[i][0])] for i in range(num_layers)]

        # Step 2: apply channel (in-filter) permutations
        in_filter_permuted_embeddings = [
            np.array([filter_embeddings[inverse_permutation] for filter_embeddings, inverse_permutation in
                      zip(filter_permuted_embeddings[i], np.argsort(permutations[i][1]))]) for i in
            range(num_layers)]

        return [torch.Tensor(in_filter_permuted_embeddings[i]).reshape(embeddings[i].shape).to(embeddings[i].device)
                for i in range(len(embeddings))]
