from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from HAND.models.model import OriginalModel
from HAND.tasks.cifar_resnet import _weights_init
from HAND.tsp.tsp import get_max_sim_order


class PermutedResNet(OriginalModel):

    def __init__(self, block, num_blocks, num_classes=10):
        super(PermutedResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.feature_maps = []
        self.apply(_weights_init)
        original_learnable_weights = self.get_original_learnable_weights()
        self.num_hidden = [[weight.shape[0], weight.shape[1]] for weight in original_learnable_weights]
        self.kernel_sizes = [layer_weights.shape[-1] for layer_weights in original_learnable_weights]
        self.max_sim_order = None
        self.inverse_permutation = None

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def layer_forward(self, layer, x):
        activations = []
        for block in layer:
            x, block_activations = block(x)
            activations.extend(block_activations)
        return x, activations

    def forward(self, x, extract_feature_maps=False):
        activations = []
        out = self.bn1(self.conv1(x))
        activations.append(out)
        out = F.relu(out)
        out, layer_activations = self.layer_forward(self.layer1, out)
        activations.extend(layer_activations)
        out, layer_activations = self.layer_forward(self.layer2, out)
        activations.extend(layer_activations)
        out, layer_activations = self.layer_forward(self.layer3, out)
        activations.extend(layer_activations)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if extract_feature_maps:
            self.feature_maps = activations
        return out

    def get_feature_maps(self, batch: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self.feature_maps = []
        output = self.forward(batch, extract_feature_maps=True)
        return output, self.feature_maps

    def get_fully_connected_weights(self) -> torch.Tensor:
        return self.linear.weight

    def get_original_learnable_weights(self) -> List[torch.Tensor]:
        learnable_weights = [self.conv1.weight]
        for layer in [self.layer1, self.layer2, self.layer3]:
            layer_learanable_weights = []
            for block in layer:
                learnable_weights.extend(block.get_learnable_weights())
                layer_learanable_weights.extend(block.get_learnable_weights())
        return learnable_weights

    def get_learnable_weights(self) -> List[torch.Tensor]:
        learnable_weights = self.get_original_learnable_weights()
        num_layers = len(learnable_weights)
        original_weights_shapes = [learnable_weights[i].shape for i in range(num_layers)]
        return [learnable_weights[i].view(-1, self.kernel_sizes[i], self.kernel_sizes[i]).permute(
            self.max_sim_order[i]).view(original_weights_shapes[i]) for i in range(num_layers)]

    def calculate_permutation(self):
        self.max_sim_order = [
            get_max_sim_order(layer_weights.detach().cpu().numpy().reshape((-1, self.kernel_sizes[i] ** 2)), False)
            for
            i, layer_weights in
            enumerate(self.get_original_learnable_weights())]
        self.inverse_permutation = self.calculate_inverse_permutation(self.max_sim_order)

    def calculate_inverse_permutation(self, permutation: List[np.array]):
        return [np.argsort(permutation[i]) for i in range(len(permutation))]
