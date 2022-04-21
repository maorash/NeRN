from typing import List, Tuple

import torch
from torch import nn as nn
from torch.nn import functional as F

from HAND.models.model import OriginalModel, ReconstructedModel
from HAND.positional_embedding import get_positional_encoding_embedder


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

    def get_feature_maps(self, batch: torch.TensorType) -> List[torch.TensorType]:
        self.feature_maps = []
        self.forward(batch, extract_feature_maps=True)
        return self.feature_maps

    def get_learnable_weights(self):
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
    def __init__(self, original_model: SimpleNet):
        super().__init__(original_model)
        self.positional_embeddings = self._calculate_positional_embeddings()

    def get_positional_embeddings(self) -> Tuple[List[Tuple], List[torch.TensorType]]:
        return self.positional_embeddings

    def _calculate_position_embeddings(self) -> Tuple[List[Tuple], List[torch.TensorType]]:
        indices = []
        for filter_idx in range(self.original_model.num_hidden):
            indices.append((0, filter_idx, 0))  # Layer 0, Filter i, Channel 0 (in_channels=1)

        for layer_idx in range(1, len(self.original_model.get_learnable_weights())):
            for filter_idx in range(self.original_model.num_hidden):
                for channel_idx in range(self.original_model.num_hidden):
                    indices.append((layer_idx, filter_idx, channel_idx))

        pe_calculator, output_dim = get_positional_encoding_embedder(3)  # TODO implement pos_emb correctly
        positional_embeddings = pe_calculator.embed(indices)

        return indices, positional_embeddings

    def update_weights(self, index: Tuple, weight: torch.TensorType):
        pass
