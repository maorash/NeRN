from typing import List, Tuple

import torch
from torch import nn as nn
from torch.nn import functional as F

from HAND.models.model import OriginalModel, ReconstructedModel
from HAND.options import EmbeddingsConfig
from HAND.positional_embedding import MyPositionalEncoding


class VGG8(OriginalModel):
    def __init__(self, input_size=28, input_channels=3, num_classes: int = 10, **kwargs):
        super(VGG8, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.num_hidden = [6, 16, 32, 64, 64]
        self.conv_layer_channels = [self.input_channels] + self.num_hidden
        self.num_conv_layers = len(self.conv_layer_channels)
        self.pooling_factor = 2 ** (self.num_conv_layers-1)
        self.fc_layer_channels = [(self.input_size // self.pooling_factor) ** 2 * self.conv_layer_channels[-1], 128,
                                  128, self.num_classes]
        self.num_fc_layers = len(self.fc_layer_channels)
        self.layers_list = [
            nn.Conv2d(self.conv_layer_channels[i], self.conv_layer_channels[i + 1], (3, 3), (1, 1), padding='same') for
            i in range(self.num_conv_layers - 1)]
        self.conv_layers = nn.ModuleList(self.layers_list)

        self.fc_layers = [nn.Linear(self.fc_layer_channels[i], self.fc_layer_channels[i + 1]) for i in
                          range(self.num_fc_layers - 1)]
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.feature_maps = []

    def get_feature_maps(self, batch: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self.feature_maps = []
        output = self.forward(batch, extract_feature_maps=True)
        return output, self.feature_maps

    def get_learnable_weights(self):
        tensors = []
        for conv_layer in self.conv_layers:
            tensors.append(conv_layer.weight)
        return tensors

    def get_fully_connected_weights(self) -> List[torch.Tensor]:
        tensors = []
        for fc_layer in self.fc_layers:
            tensors.append(fc_layer.weight)
        return tensors

    def forward(self, x, extract_feature_maps=False):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            if extract_feature_maps:
                self.feature_maps.append(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        return x


class ReconstructedVGG8(ReconstructedModel):
    def __init__(self, original_model: VGG8, embeddings_cfg: EmbeddingsConfig, sampling_mode: str = None):
        super().__init__(original_model, sampling_mode)
        self.indices = self._get_tensor_indices()
        self.positional_encoder = MyPositionalEncoding(embeddings_cfg)
        self.positional_embeddings = self._calculate_position_embeddings()

    def _get_tensor_indices(self) -> List[List[Tuple]]:
        indices = []
        for layer_idx in range(0, len(self.original_model.get_learnable_weights())):
            curr_layer_indices = []
            for filter_idx in range(self.original_model.num_hidden[layer_idx]):
                for channel_idx in range(self.original_model.conv_layer_channels[layer_idx]):
                    curr_layer_indices.append((layer_idx, filter_idx, channel_idx))
            indices.append(curr_layer_indices)

        return indices
