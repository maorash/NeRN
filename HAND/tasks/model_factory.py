import torch

from HAND.tasks.simple_net import SimpleNet, ReconstructedSimpleNet3x3
from HAND.tasks.vgg8 import VGG8, ReconstructedVGG83x3
from HAND.tasks.resnet18 import ResNet18, ReconstructedResNet183x3
from HAND.tasks.resnet14 import ResNet14, ReconstructedResNet143x3
from HAND.options import TrainConfig


class ModelFactory:
    models = {
        "SimpleNet": (SimpleNet, ReconstructedSimpleNet3x3),
        "VGG8": (VGG8, ReconstructedVGG83x3),
        "ResNet18": (ResNet18, ReconstructedResNet183x3),
        "ResNet14": (ResNet14, ReconstructedResNet143x3)
    }

    @staticmethod
    def get(cfg: TrainConfig, device: torch.device, **kwargs):
        if cfg.task.original_model_name not in ModelFactory.models:
            raise ValueError("Unsupported original model name")

        model = ModelFactory.models[cfg.task.original_model_name][0](**kwargs).to(device)
        model.load_state_dict(torch.load(cfg.original_model_path, map_location=device))
        reconstructed_model = ModelFactory.models[cfg.task.original_model_name][1](model, cfg.hand.embeddings).to(device)

        return model, reconstructed_model

