import random
from abc import ABC, abstractmethod
from typing import List, Tuple

import einops
import numpy as np
import torch
from torch import nn, distributions

from NeRN.options import NeRNConfig
from NeRN.predictors.activations import ActivationsFactory


class NeRNPredictorBase(nn.Module, ABC):
    def __init__(self, cfg: NeRNConfig, input_size: int):
        super().__init__()
        self.cfg = cfg
        self.input_size = input_size
        self.act_layer = ActivationsFactory.get(cfg.act_layer)
        # Layer index for computing gradients (for when cfg.weights_batch_method is a layer-based method)
        self.layer_ind_for_grads = 0
        # Random batch variables for computing gradients (for when cfg.weights_batch_method is a batch-based method)
        self.permuted_positional_embeddings = None
        self.random_batch_idx = 0

    def save(self, path: str):
        torch.save(self, path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path).state_dict())

    @abstractmethod
    def forward(self, positional_embedding: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def output_size(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def calc_weights_norms(self, original_weights: List[torch.Tensor]) -> List[torch.Tensor]:
        raise NotImplementedError()

    @staticmethod
    def predict_all(predictor: 'NeRNPredictorBase', positional_embeddings: List[torch.Tensor],
                    original_weights: List[torch.Tensor], learnable_weights_shapes: List[torch.Size]) -> List[torch.Tensor]:
        predicted_weights_shapes = [(layer_shape[0], layer_shape[1], predictor.output_size, predictor.output_size) for
                                    layer_shape
                                    in learnable_weights_shapes]
        reconstructed_weights = []
        if predictor.cfg.weights_batch_method == 'all':
            for embedding, shape in zip(positional_embeddings, predicted_weights_shapes):
                layer_reconstructed_weights = NeRNPredictorBase._predict_weights(predictor, embedding).reshape(shape)
                layer_reconstructed_weights.retain_grad()
                reconstructed_weights.append(layer_reconstructed_weights)
        elif predictor.cfg.weights_batch_method in ('sequential_layer', 'random_layer'):
            reconstructed_weights = NeRNPredictorBase._predict_all_by_layers(predictor,
                                                                             positional_embeddings,
                                                                             predicted_weights_shapes)
        elif predictor.cfg.weights_batch_method in ('random_batch', 'random_batch_without_replacement',
                                                    'random_weighted_batch'):
            reconstructed_weights = NeRNPredictorBase._predict_all_by_random_batches(predictor,
                                                                                     positional_embeddings,
                                                                                     original_weights,
                                                                                     predicted_weights_shapes)
        else:
            raise ValueError("Unsupported predictor method")

        return reconstructed_weights

    @staticmethod
    def _predict_all_by_random_batches(predictor: 'NeRNPredictorBase', positional_embeddings: List[torch.Tensor],
                                       original_weights: List[torch.Tensor], predicted_weights_shapes: List[Tuple]) \
            -> List[torch.Tensor]:
        weights_norms = predictor.calc_weights_norms(original_weights)
        stacked_embeddings = torch.vstack(positional_embeddings)
        if predictor.cfg.weights_batch_method == 'random_batch':
            predictor.permuted_positional_embeddings = torch.randperm(stacked_embeddings.shape[0],
                                                                      device=stacked_embeddings.device)
            indices_with_grads = predictor.permuted_positional_embeddings[:predictor.cfg.weights_batch_size]
            indices_without_grads = predictor.permuted_positional_embeddings[predictor.cfg.weights_batch_size:]
        elif predictor.cfg.weights_batch_method == 'random_weighted_batch':
            stacked_norms = torch.concat(weights_norms)
            if random.uniform(0, 1) < 0.8:
                predictor.permuted_indices = torch.randperm(stacked_embeddings.shape[0],
                                                            device=stacked_embeddings.device)
                indices_with_grads = predictor.permuted_indices[:predictor.cfg.weights_batch_size]
                indices_without_grads = predictor.permuted_indices[predictor.cfg.weights_batch_size:]
            else:
                indices_with_grads = distributions.Categorical(stacked_norms).sample([predictor.cfg.weights_batch_size])
                indices_without_grads = set(range(stacked_embeddings.shape[0])) - set(indices_with_grads.tolist())
                indices_without_grads = torch.Tensor(list(indices_without_grads)).to(
                    device=stacked_embeddings.device).long()
        else:
            num_batches = -(stacked_embeddings.shape[0] // -predictor.cfg.weights_batch_size)
            if predictor.permuted_positional_embeddings is None or predictor.random_batch_idx >= num_batches:
                predictor.permuted_positional_embeddings = torch.randperm(stacked_embeddings.shape[0],
                                                                          device=stacked_embeddings.device)
            with_grads_ind_begin = predictor.random_batch_idx * predictor.cfg.weights_batch_size
            with_grads_ind_end = (predictor.random_batch_idx + 1) * predictor.cfg.weights_batch_size
            indices_with_grads = predictor.permuted_positional_embeddings[with_grads_ind_begin: with_grads_ind_end]
            indices_without_grads = torch.concat([predictor.permuted_positional_embeddings[:with_grads_ind_begin],
                                                  predictor.permuted_positional_embeddings[with_grads_ind_end:]])
            predictor.random_batch_idx = (predictor.random_batch_idx + 1) % num_batches
        predicted_with_grads = NeRNPredictorBase._predict_weights(predictor, stacked_embeddings[indices_with_grads])
        predicted_with_grads.retain_grad()
        predicted = torch.zeros(stacked_embeddings.shape[0], predicted_with_grads.shape[1],
                                device=stacked_embeddings.device)
        predicted[indices_with_grads] = predicted_with_grads
        if indices_without_grads.shape[0] > 0:
            with torch.no_grad():
                predicted_without_grads = NeRNPredictorBase._predict_weights(predictor,
                                                                             stacked_embeddings[indices_without_grads])
                predicted[indices_without_grads] = predicted_without_grads
        reconstructed_weights = torch.vsplit(predicted,
                                             list(np.cumsum([pe.shape[0] for pe in positional_embeddings])))[:-1]
        reconstructed_weights = [w.reshape(s) for w, s in zip(reconstructed_weights, predicted_weights_shapes)]
        return reconstructed_weights

    @staticmethod
    def _predict_all_by_layers(predictor: 'NeRNPredictorBase', positional_embeddings: List[torch.Tensor],
                               predicted_weights_shapes: List[Tuple]) -> List[torch.Tensor]:
        reconstructed_weights = []
        predictor.layer_ind_for_grads = (predictor.layer_ind_for_grads + 1) % len(positional_embeddings) if \
            predictor.cfg.weights_batch_method == 'sequential_layer' else \
            torch.randint(0, len(positional_embeddings), (1,)).item()
        for layer_ind, (embedding, shape) in enumerate(zip(positional_embeddings, predicted_weights_shapes)):
            if layer_ind == predictor.layer_ind_for_grads:
                layer_reconstructed_weights = NeRNPredictorBase._predict_weights(predictor, embedding).reshape(shape)
                layer_reconstructed_weights.retain_grad()
            else:
                with torch.no_grad():
                    layer_reconstructed_weights = NeRNPredictorBase._predict_weights(predictor, embedding).reshape(shape)
            reconstructed_weights.append(layer_reconstructed_weights)

        return reconstructed_weights

    @staticmethod
    def _predict_weights(predictor: 'NeRNPredictorBase', layer_positional_embeddings):
        if predictor.cfg.weights_batch_size is None:
            return predictor.forward(layer_positional_embeddings)

        layer_reconstructed_weights = []
        for batch_idx in range(0, layer_positional_embeddings.shape[0], predictor.cfg.weights_batch_size):
            weights_batch = layer_positional_embeddings[batch_idx: batch_idx + predictor.cfg.weights_batch_size]
            layer_reconstructed_weights.append(predictor.forward(weights_batch))
        layer_reconstructed_weights = torch.vstack(layer_reconstructed_weights)

        return layer_reconstructed_weights


class NeRNKxKPredictor(NeRNPredictorBase):
    """
    Given 3 positional embeddings: (Layer, Filter, Channel) returns a KxK filter tensor
    """

    def __init__(self, cfg: NeRNConfig, input_size: int):
        super().__init__(cfg, input_size)
        self.hidden_size = cfg.hidden_layer_size
        self.layers = self._construct_layers()
        self.final_linear_layer = nn.Linear(self.hidden_size, cfg.output_size ** 2)

    def _construct_layers(self):
        blocks = [nn.Linear(self.input_size, self.hidden_size)]
        blocks.extend([nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.cfg.num_blocks - 2)])
        return nn.ModuleList(blocks)

    def forward(self, positional_embedding: torch.Tensor) -> torch.Tensor:
        x = positional_embedding
        for layer in self.layers:
            x = layer(x)
            x = self.act_layer(x)
        x = self.final_linear_layer(x)
        return x

    @property
    def output_size(self) -> int:
        return self.cfg.output_size

    def calc_weights_norms(self, weights: List[torch.Tensor]) -> List[torch.Tensor]:
        return [torch.norm(einops.rearrange(weight, 'cout cin h w -> (cout cin) (h w)'), dim=1)
                for weight in weights]


class NeRNKxKNerFPredictor(NeRNKxKPredictor):
    """
    Given 3 positional embeddings: (Layer, Filter, Channel) returns a KxK filter tensor
    """

    def __init__(self, cfg: NeRNConfig, input_size: int):
        self.skips = [self.cfg.num_blocks // 2]  # nerf uses a skip in the middle of the blocks
        super().__init__(cfg, input_size)

    def _construct_layers(self):
        blocks = [nn.Linear(self.input_size, self.hidden_size)]
        for i in range(1, self.cfg.num_blocks - 1):
            if i in self.skips:
                layer = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
            else:
                layer = nn.Linear(self.hidden_size, self.hidden_size)
            blocks.append(layer)
        return nn.ModuleList(blocks)

    def forward(self, positional_embedding: torch.Tensor) -> torch.Tensor:
        x = positional_embedding
        for i, layer in enumerate(self.layers):
            if i in self.skips:
                x = layer(torch.cat([positional_embedding, x], -1))
            else:
                x = layer(x)
            x = self.act_layer(x)
        x = self.final_linear_layer(x)
        return x


class NeRNKxKResidualPredictor(NeRNKxKPredictor):
    """
    Given 3 positional embeddings: (Layer, Filter, Channel) returns a KxK filter tensor
    """

    def __init__(self, cfg: NeRNConfig, input_size: int):
        self.skips = [self.cfg.num_blocks // 2]  # nerf uses a skip in the middle of the blocks
        super().__init__(cfg, input_size)

    def _construct_layers(self):
        blocks = [nn.Linear(self.input_size, self.hidden_size)]
        for i in range(1, self.cfg.num_blocks - 1):
            layer = nn.Linear(self.hidden_size, self.hidden_size)
            blocks.append(layer)
        return nn.ModuleList(blocks)

    def forward(self, positional_embedding: torch.Tensor) -> torch.Tensor:
        x = self.layers[0](positional_embedding)
        first_layer_out = x
        for i in range(1, len(self.layers)):
            x = self.layers[i](x)
            if i in self.skips:
                x += first_layer_out
            x = self.act_layer(x)
        x = self.final_linear_layer(x)
        return x


class NeRNBasicPredictor(NeRNPredictorBase):
    """
    Given 5 positional embeddings: (Layer, Filter, Channel, Height, Width) returns a single floating point
    """

    @property
    def output_size(self) -> int:
        return 1

    def forward(self, positional_embedding: List[torch.Tensor]) -> List[torch.Tensor]:
        pass
