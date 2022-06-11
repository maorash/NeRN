from typing import List, Tuple

import torch
from torch import nn as nn
from torch.nn import functional as F

from HAND.models.model import OriginalModel, ReconstructedModel
from HAND.positional_embedding import MyPositionalEncoding


class SimpleNet(OriginalModel):
    def __init__(self, num_hidden=32, num_layers=3):
        super(SimpleNet, self).__init__()
        self.num_hidden = num_hidden
        self.layers_list = [nn.Conv2d(1, num_hidden, (3, 3), (1, 1))]
        self.layers_list.extend([nn.Conv2d(num_hidden, num_hidden, (3, 3), (1, 1)) for _ in range(num_layers - 1)])
        self.convs = nn.ModuleList(self.layers_list)
        self.dropout1 = nn.Dropout(0.25)
        self.fc = nn.Linear(num_hidden * 11 * 11, 10)
        self.feature_maps = []

    def get_feature_maps(self, batch: torch.Tensor) -> List[torch.Tensor]:
        self.feature_maps = []
        self.forward(batch, extract_feature_maps=True)
        return self.feature_maps

    def get_learnable_weights(self):  # TODO: implement index accessing
        tensors = []
        for conv in self.convs:
            tensors.append(conv.weight)
        return tensors

    def forward(self, x, extract_feature_maps=False):
        for layer_idx in range(len(self.convs)):
            x = self.convs[layer_idx](x)
            if extract_feature_maps:
                self.feature_maps.append(x)
            x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


class ReconstructedSimpleNet3x3(ReconstructedModel):
    def __init__(self, original_model: SimpleNet, **positional_encoding_args):
        super().__init__(original_model)
        self.indices = self._get_tensor_indices()
        self.positional_encoder = MyPositionalEncoding(**positional_encoding_args)
        self.positional_embeddings = self._calculate_position_embeddings()

    def _get_tensor_indices(self) -> List[Tuple]:
        indices = []
        for filter_idx in range(self.original_model.num_hidden):
            indices.append((0, filter_idx, 0))  # Layer 0, Filter i, Channel 0 (in_channels=1)

        for layer_idx in range(1, len(self.original_model.get_learnable_weights())):
            for filter_idx in range(self.original_model.num_hidden):
                for channel_idx in range(self.original_model.num_hidden):
                    indices.append((layer_idx, filter_idx, channel_idx))
        return indices

    def _calculate_position_embeddings(self) -> List[torch.Tensor]:
        # TODO: is this best?
        positional_embeddings = [self.positional_encoder(idx) for idx in self.indices]
        return positional_embeddings

    def get_indices_and_positional_embeddings(self) -> Tuple[List[Tuple], List[torch.Tensor]]:
        return self.indices, self.positional_embeddings

    def aggregate_predicted_weights(self, predicted_weights_raw: List[torch.Tensor]) -> List[torch.Tensor]:
        new_weights = [torch.zeros_like(conv.weight) for conv in self.original_model.convs]
        for idx, weight in zip(self.indices, predicted_weights_raw):
            i, j, k = idx
            new_weights[i][j, k] = weight.reshape(3, 3)
        return new_weights
