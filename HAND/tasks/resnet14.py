import pickle
from typing import List, Tuple

import torch
import torchvision.models

from HAND.models.model import OriginalModel, ReconstructedModel
from HAND.options import EmbeddingsConfig
from HAND.positional_embedding import MyPositionalEncoding


class ResNet14(OriginalModel):
    def __init__(self, num_classes: int = 10, **kwargs):
        super(ResNet14, self).__init__()
        self.num_hidden = [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256]
        self.input_channels = 3
        self.model = torchvision.models.resnet18(pretrained=False)
        # self.layers_names = ['layer1', 'layer2', 'layer3', 'layer4']
        self.layers_names = ['layer1', 'layer2', 'layer3']
        self.feature_maps = []
        self.num_classes = num_classes
        self.model.fc = torch.nn.Linear(self.num_hidden[-1], self.num_classes)

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

    def layer_forward(self, layer: torch.nn.Sequential, x, extract_feature_maps=False):
        activations = []
        for block in layer:
            x, block_activations = self.block_forward(block, x, extract_feature_maps)
            activations += block_activations
        return x, activations

    def forward(self, x, extract_feature_maps=True):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x, layer1_activations = self.layer_forward(self.model.layer1, x, extract_feature_maps)
        x, layer2_activations = self.layer_forward(self.model.layer2, x, extract_feature_maps)
        x, layer3_activations = self.layer_forward(self.model.layer3, x, extract_feature_maps)
        # x, layer4_activations = self.layer_forward(self.model.layer4, x, extract_feature_maps)
        # activations = layer1_activations + layer2_activations + layer3_activations + layer4_activations
        activations = layer1_activations + layer2_activations + layer3_activations
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        if extract_feature_maps is True:
            self.feature_maps = activations
        return x


class ReconstructedResNet143x3(ReconstructedModel):
    def __init__(self, original_model: ResNet14, embeddings_cfg: EmbeddingsConfig):
        super().__init__(original_model)
        self.embeddings_cfg = embeddings_cfg
        self.normalized_indices = None
        self.indices = self._get_tensor_indices()
        self.positional_encoder = MyPositionalEncoding(embeddings_cfg)
        self.positional_embeddings = self._calculate_position_embeddings()

    def _get_indices_boundaries(self) -> List[List[int]]:
        weights_shapes = self.get_learnable_weights_shapes()
        layer_boundaries, filter_boundaries, channel_boundaries = [], [], []
        for layer_weights_shape in weights_shapes:
            layer_boundaries.append(len(weights_shapes))
            filter_boundaries.append(layer_weights_shape[0])
            channel_boundaries.append(layer_weights_shape[1])
        return [layer_boundaries, filter_boundaries, channel_boundaries]

    def _get_tensor_indices(self) -> List[List[Tuple]]:
        indices = []
        normalize_indices = []

        max_index = max([max(weights_shape) for weights_shape in self.get_learnable_weights_shapes()])

        num_layers = len(self.original_model.get_learnable_weights())
        for layer_idx in range(0, num_layers):
            curr_layer_indices = []
            curr_normalized_layer_indices = []
            curr_num_filters = self.original_model.num_hidden[layer_idx + 1]
            for filter_idx in range(curr_num_filters):
                curr_num_channels = self.original_model.num_hidden[layer_idx]
                for channel_idx in range(curr_num_channels):
                    curr_layer_indices.append((layer_idx, filter_idx, channel_idx))
                    if self.embeddings_cfg.normalization_mode is None:
                        curr_normalized_layer_indices.append((layer_idx, filter_idx, channel_idx))
                    elif self.embeddings_cfg.normalization_mode == "global":
                        curr_normalized_layer_indices.append((layer_idx / max_index, filter_idx / max_index, channel_idx / max_index))
                    elif self.embeddings_cfg.normalization_mode == "local":
                        curr_normalized_layer_indices.append((layer_idx / num_layers, filter_idx / curr_num_filters, channel_idx / curr_num_channels))
                    else:
                        raise ValueError(f"Unsupported normalization mode {self.normalization_mode}")

            indices.append(curr_layer_indices)
            normalize_indices.append(curr_normalized_layer_indices)

        self.normalized_indices = normalize_indices

        return indices

    def _calculate_position_embeddings(self) -> List[List[torch.Tensor]]:
        embeddings_cache_filename = f"{__name__}_embeddings_{hash(self.positional_encoder)}.pkl"
        try:
            print("Trying to load precomputed embeddings")
            with open(embeddings_cache_filename, "rb") as f:
                positional_embeddings = pickle.load(f)
            print("Loaded precomputed embeddings")
            return positional_embeddings
        except Exception:
            print("Couldn't load precomputed embeddings, hang on tight..")

        positional_embeddings = []
        for i, layer_indices in enumerate(self.normalized_indices):
            print(f"Calculating layer {i}/{len(self.normalized_indices)} embeddings. It gets slower")
            layer_embeddings = []
            for idx in layer_indices:
                layer_embeddings.append(self.positional_encoder(idx))
            positional_embeddings.append(layer_embeddings)

        with open(embeddings_cache_filename, "wb") as f:
            pickle.dump(positional_embeddings, f)
            print("Saved computed embeddings")

        return positional_embeddings

    def get_indices_and_positional_embeddings(self) -> Tuple[List[List[Tuple]], List[List[torch.Tensor]]]:
        return self.indices, self.positional_embeddings
