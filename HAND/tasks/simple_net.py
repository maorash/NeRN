from typing import List, Tuple

import torch
from torch import nn as nn
from torch.nn import functional as F

from HAND.options import EmbeddingsConfig
from HAND.models.model import OriginalModel, ReconstructedModel
from HAND.positional_embedding import MyPositionalEncoding


class SimpleNet(OriginalModel):
    def __init__(self, num_hidden=None, num_layers=3, input_size=28, input_channels=3, **kwargs):
        super(SimpleNet, self).__init__()
        self.input_channels = input_channels
        self.num_hidden = num_hidden if num_hidden is not None else [32] * num_layers
        self.input_size = input_size
        self.layers_list = [nn.Conv2d(self.input_channels, self.num_hidden[0], (3, 3), (1, 1), padding='same')]
        self.layers_list.extend(
            [nn.Conv2d(self.num_hidden[i], self.num_hidden[i + 1], (3, 3), (1, 1), padding='same') for i in
             range(num_layers - 1)])
        self.convs = nn.ModuleList(self.layers_list)
        self.dropout1 = nn.Dropout(0.25)
        self.fc = nn.Linear(self.num_hidden[-1] * input_size // 2 * input_size // 2, 10)
        self.feature_maps = []

    def get_feature_maps(self, batch: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self.feature_maps = []
        output = self.forward(batch, extract_feature_maps=True)
        return output, self.feature_maps

    def get_learnable_weights(self):  # TODO: implement index accessing
        tensors = []
        for conv in self.convs:
            tensors.append(conv.weight)
        return tensors

    def get_fully_connected_weights(self) -> torch.Tensor:
        return self.fc.weight

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
    def __init__(self, original_model: SimpleNet, embeddings_cfg: EmbeddingsConfig):
        super().__init__(original_model)
        self.indices = self._get_tensor_indices()
        self.positional_encoder = MyPositionalEncoding(embeddings_cfg)
        self.positional_embeddings = self._calculate_position_embeddings()

    def _get_tensor_indices(self) -> List[List[Tuple]]:
        indices = []
        num_channels_in_layers = [self.original_model.input_channels] + self.original_model.num_hidden

        for layer_idx in range(len(self.original_model.get_learnable_weights())):
            curr_layer_indices = []
            for filter_idx in range(self.original_model.num_hidden[layer_idx]):
                for channel_idx in range(num_channels_in_layers[layer_idx]):
                    curr_layer_indices.append((layer_idx, filter_idx, channel_idx))
            indices.append(curr_layer_indices)

        return indices