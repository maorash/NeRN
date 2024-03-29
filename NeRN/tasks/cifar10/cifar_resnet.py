'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

from NeRN.models.model import OriginalModel


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(3, 3), stride=(stride, stride), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.option = option
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'cifar10':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=(1, 1), stride=(stride, stride),
                              bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
        else:
            self.option = "C"

    def forward(self, x):
        activations = []
        out = self.bn1(self.conv1(x))
        activations.append(out)
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        activations.append(out)
        out += self.shortcut(x)
        activations.append(out)
        out = F.relu(out)
        return out, activations

    def get_learnable_weights(self):
        if self.option == 'A' or self.option == 'C':
            return [self.conv1.weight, self.conv2.weight]
        else:
            return [self.conv1.weight, self.conv2.weight, self.shortcut[0].weight]


class BasicBlockA(BasicBlock):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockA, self).__init__(in_planes, planes, stride, option='A')


class BasicBlockB(BasicBlock):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockB, self).__init__(in_planes, planes, stride, option='cifar10')


class ResNet(OriginalModel):

    def __init__(self, block, num_blocks, num_classes=10, **kwargs):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.feature_maps = []
        self.apply(_weights_init)
        self.num_hidden = [[weight.shape[0], weight.shape[1]] for weight in self.get_learnable_weights()]

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

    def get_learnable_weights(self) -> List[torch.Tensor]:
        learnable_weights = [self.conv1.weight]
        for layer in [self.layer1, self.layer2, self.layer3]:
            layer_learanable_weights = []
            for block in layer:
                learnable_weights.extend(block.get_learnable_weights())
                layer_learanable_weights.extend(block.get_learnable_weights())
        return learnable_weights


def resnet20(basic_block_option='A', **kwargs):
    basic_block = BasicBlockA if basic_block_option == 'A' else BasicBlockB
    return ResNet(basic_block, [3, 3, 3], **kwargs)


def resnet32(basic_block_option='A', **kwargs):
    basic_block = BasicBlockA if basic_block_option == 'A' else BasicBlockB
    return ResNet(basic_block, [5, 5, 5], **kwargs)


def resnet44(basic_block_option='A', **kwargs):
    basic_block = BasicBlockA if basic_block_option == 'A' else BasicBlockB
    return ResNet(basic_block, [7, 7, 7], **kwargs)


def resnet56(basic_block_option='A', **kwargs):
    basic_block = BasicBlockA if basic_block_option == 'A' else BasicBlockB
    return ResNet(basic_block, [9, 9, 9], **kwargs)


def resnet110(basic_block_option='A', **kwargs):
    basic_block = BasicBlockA if basic_block_option == 'A' else BasicBlockB
    return ResNet(basic_block, [18, 18, 18], **kwargs)


def resnet1202(basic_block_option='A', **kwargs):
    basic_block = BasicBlockA if basic_block_option == 'A' else BasicBlockB
    return ResNet(basic_block, [200, 200, 200], **kwargs)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
