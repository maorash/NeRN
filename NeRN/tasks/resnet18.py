from typing import List, Tuple

import torch
import torchvision.models

from NeRN.models.model import OriginalModel, ReconstructedModel
from NeRN.options import Config


class ResNet18(OriginalModel):
    def __init__(self, num_classes: int = 10, **kwargs):
        super(ResNet18, self).__init__()
        self.num_hidden = [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
        self.input_channels = 3
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
        # tensors.append(self.model.conv1.weight)
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
            activations.append(out)
        out = block.relu(out)

        out = block.conv2(out)
        out = block.bn2(out)
        if extract_feature_maps is True:
            activations.append(out)

        if block.downsample is not None:
            identity = block.downsample(x)

        out += identity
        if extract_feature_maps is True:
            activations.append(out)
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
        first_activation = [x]
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x, layer1_activations = self.layer_forward(self.model.layer1, x, extract_feature_maps)
        x, layer2_activations = self.layer_forward(self.model.layer2, x, extract_feature_maps)
        x, layer3_activations = self.layer_forward(self.model.layer3, x, extract_feature_maps)
        x, layer4_activations = self.layer_forward(self.model.layer4, x, extract_feature_maps)
        activations = first_activation + layer1_activations + layer2_activations + layer3_activations + layer4_activations
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        if extract_feature_maps is True:
            self.feature_maps = activations
        return x

    def load(self, path: str, device: torch.device):
        self.model.load_state_dict(torch.load(path, map_location=device))


class ReconstructedResNet18(ReconstructedModel):
    def __init__(self, original_model: ResNet18, train_cfg: Config, device: str, sampling_mode: str = None):
        super().__init__(original_model, train_cfg, device, sampling_mode)

    def __str__(self):
        return f"{type(self).__name__}"
