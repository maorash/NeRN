import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Any, List, Tuple

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']

from ..models.model import OriginalModel, ReconstructedModel
from ..options import Config

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth',
}


class Fire(nn.Module):

    def __init__(
            self,
            inplanes: int,
            squeeze_planes: int,
            expand1x1_planes: int,
            expand3x3_planes: int
    ) -> None:
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, extract_feature_maps=True) -> (torch.Tensor, List[torch.Tensor]):
        features = []

        x = self.squeeze(x)
        if extract_feature_maps:
            features.append(x)
        x = self.squeeze_activation(x)

        x1 = self.expand1x1(x)
        x3 = self.expand3x3(x)
        if extract_feature_maps:
            features.append(x1)
            features.append(x3)

        return torch.cat([
            self.expand1x1_activation(x1),
            self.expand3x3_activation(x3)
        ], 1), features


class SqueezeNet(OriginalModel):

    def __init__(
            self,
            version: str = '1_1',
            num_classes: int = 1000
    ) -> None:
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        self.feature_maps = []

    def forward(self, x: torch.Tensor, extract_feature_maps=True) -> torch.Tensor:
        features = []
        for m in self.features:
            if extract_feature_maps:
                if isinstance(m, Fire):
                    x, subfeatures = m(x, extract_feature_maps=True)
                    features.extend(subfeatures)
                elif isinstance(m, nn.Conv2d):
                    x = m(x)
                    features.append(x)
                else:
                    x = m(x)
            else:
                x = m(x)

        x = self.classifier(x)
        if extract_feature_maps:
            self.feature_maps = features

        return torch.flatten(x, 1)

    def load(self, path: str, device: torch.device):
        self.load_state_dict(torch.load(path, map_location=device))

    def get_feature_maps(self, batch: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self.feature_maps = []
        output = self.forward(batch, extract_feature_maps=True)
        return output, self.feature_maps

    def get_learnable_weights(self):
        tensors = []
        for m in self.features:
            if isinstance(m, Fire):
                tensors.append(m.squeeze.weight)
                tensors.append(m.expand1x1.weight)
                tensors.append(m.expand3x3.weight)
            if isinstance(m, nn.Conv2d):
                tensors.append(m.weight)

        return tensors

    def get_fully_connected_weights(self) -> List[torch.Tensor]:
        tensors = [self.classifier[2].weight]
        return tensors


class ReconstructedSqueezeNet(ReconstructedModel):
    def __init__(self, original_model: SqueezeNet, train_cfg: Config, device: str, sampling_mode: str = None):
        super().__init__(original_model, train_cfg, device, sampling_mode)

    def __str__(self):
        return f"{type(self).__name__}"

