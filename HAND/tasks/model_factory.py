import torch

from HAND.tasks.simple_net import SimpleNet, ReconstructedSimpleNet
from HAND.tasks.vgg8 import VGG8, ReconstructedVGG8
from HAND.tasks.resnet18 import ResNet18, ReconstructedResNet18
from HAND.tasks.resnet14 import ResNet14, ReconstructedResNet14, ReconstructedPermutedResNet143x3
from HAND.options import TrainConfig


class ModelFactory:
    models = {
        "SimpleNet": (SimpleNet, ReconstructedSimpleNet),
        "VGG8": (VGG8, ReconstructedVGG8),
        "ResNet18": (ResNet18, ReconstructedResNet18),
        "ResNet14": (ResNet14, ReconstructedResNet14),
        "PermutedResNet14": (ResNet14, ReconstructedPermutedResNet143x3)
    }

    @staticmethod
    def get(cfg: TrainConfig, device: torch.device, **kwargs):
        if cfg.task.original_model_name not in ModelFactory.models:
            raise ValueError("Unsupported original model name")

        model = ModelFactory.models[cfg.task.original_model_name][0](**kwargs).to(device)
        model.load_state_dict(torch.load(cfg.original_model_path, map_location=device))
        reconstructed_model = ModelFactory.models[cfg.task.original_model_name][1](model, cfg.hand.embeddings,
                                                                                   sampling_mode=cfg.hand.sampling_mode).to(device)

        return model, reconstructed_model
