from typing import List, Tuple

import torch
import torchvision.models
from torch import nn as nn
from torch.nn import functional as F

from HAND.models.model import OriginalModel, ReconstructedModel
from HAND.options import EmbeddingsConfig
from HAND.positional_embedding import MyPositionalEncoding


class ResNet18(OriginalModel):
    def __init__(self, num_classes: int = 10):
        super(ResNet18, self).__init__()

        self.model = torchvision.models.resnet18(pretrained=False)
        self.layers_names = ['layer1', 'layer2', 'layer3', 'layer4']
        self.feature_maps = []
        self.num_classes = num_classes
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_classes)


    def get_feature_maps(self, batch: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self.feature_maps = []
        output = self.forward(batch, extract_feature_maps=True)
        return output, self.feature_maps

    def get_learnable_weights(self):
        tensors = []
        for layer_name in self.layers_names:
            module = self.model._modules[layer_name]
            for block in module:
                tensors.append(block.conv1.weight)
                tensors.append(block.conv2.weight)
        return tensors

    def get_fully_connected_weights(self) -> List[torch.Tensor]:
        tensors = [self.model.fc]
        return tensors

    def block_forward(self, block: torch.nn.Module, x: torch.Tensor, extract_feature_maps: bool = False) -> (
            torch.Tensor, List[torch.Tensor]):

        activations = []
        identity = x

        out = block.conv1(x)
        out = block.bn1(out)
        if extract_feature_maps is True:
            activations.append(x)
        out = block.relu(out)

        out = block.conv2(out)
        out = block.bn2(out)
        if extract_feature_maps is True:
            activations.append(x)

        if block.downsample is not None:
            identity = block.downsample(x)

        out += identity
        out = block.relu(out)

        return out, activations

    def layer_forward(self,  layer: torch.nn.Sequential, x, extract_feature_maps=False):
        activations = []
        for block in layer:
            x, block_activations = self.block_forward(block, x, extract_feature_maps)
            activations.append(block_activations)
        return x, activations

    def forward(self, x, extract_feature_maps=True):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x, layer1_activations = self.layer_forward(self.model.layer1, x, extract_feature_maps)
        x, layer2_activations = self.layer_forward(self.model.layer2, x, extract_feature_maps)
        x, layer3_activations = self.layer_forward(self.model.layer3, x, extract_feature_maps)
        x, layer4_activations = self.layer_forward(self.model.layer4, x, extract_feature_maps)
        activations = layer1_activations + layer2_activations + layer3_activations + layer4_activations
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        if extract_feature_maps is True:
            self.model.feature_maps = activations
        return x


class ReconstructedVGG3x3(ReconstructedModel):
    def __init__(self, original_model: ResNet18, embeddings_cfg: EmbeddingsConfig):
        super().__init__(original_model)
        self.indices = self._get_tensor_indices()
        self.positional_encoder = MyPositionalEncoding(embeddings_cfg)
        self.positional_embeddings = self._calculate_position_embeddings()

    def _get_tensor_indices(self) -> List[List[Tuple]]:
        indices = []
        curr_layer_indices = []
        for filter_idx in range(self.original_model.num_hidden[0]):
            curr_layer_indices.append((0, filter_idx, 0))  # Layer 0, Filter i, Channel 0 (in_channels=1)
        indices.append(curr_layer_indices)

        for layer_idx in range(1, len(self.original_model.get_learnable_weights())):
            curr_layer_indices = []
            for filter_idx in range(self.original_model.num_hidden[layer_idx]):
                for channel_idx in range(self.original_model.num_hidden[layer_idx - 1]):
                    curr_layer_indices.append((layer_idx, filter_idx, channel_idx))
            indices.append(curr_layer_indices)

        return indices

    def _calculate_position_embeddings(self) -> List[List[torch.Tensor]]:
        positional_embeddings = [[self.positional_encoder(idx) for idx in layer_indices] for layer_indices in
                                 self.indices]
        return positional_embeddings

    def get_indices_and_positional_embeddings(self) -> Tuple[List[List[Tuple]], List[List[torch.Tensor]]]:
        return self.indices, self.positional_embeddings
