import torch

from HAND.models.model import OriginalDataParallel, ReconstructedDataParallel
from HAND.tasks.simple_net import SimpleNet, ReconstructedSimpleNet
from HAND.tasks.vgg8 import VGG8, ReconstructedVGG8
from HAND.tasks.resnet18 import ResNet18, ReconstructedResNet18
from HAND.tasks.resnet14 import ResNet14, ReconstructedResNet14
from HAND.tasks.resnet20 import ResNet20, ReconstructedResNet20
from HAND.tasks.resnet56 import ResNet56, ReconstructedResNet56
from HAND.options import TrainConfig


class ModelFactory:
    models = {
        "SimpleNet": (SimpleNet, ReconstructedSimpleNet),
        "VGG8": (VGG8, ReconstructedVGG8),
        "ResNet18": (ResNet18, ReconstructedResNet18),
        "ResNet14": (ResNet14, ReconstructedResNet14),
        "ResNet20": (ResNet20, ReconstructedResNet20),
        "ResNet56": (ResNet56, ReconstructedResNet56)
    }

    @staticmethod
    def get(cfg: TrainConfig, device: torch.device, **kwargs):
        if cfg.task.original_model_name not in ModelFactory.models:
            raise ValueError("Unsupported original model name")

        model = ModelFactory.models[cfg.task.original_model_name][0](**kwargs).to(device)
        model.load(cfg.original_model_path, device)
        reconstructed_model = ModelFactory.models[cfg.task.original_model_name][1](model, cfg.hand.embeddings,
                                                                                   sampling_mode=cfg.hand.sampling_mode).to(device)
        if cfg.num_gpus > 1:
            model = OriginalDataParallel(model, device_ids=list(range(cfg.num_gpus)))
            reconstructed_model = ReconstructedDataParallel(reconstructed_model, device_ids=list(range(cfg.num_gpus)))

        return model, reconstructed_model
