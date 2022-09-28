import json
import os
import torch

from NeRN.models.model import OriginalDataParallel, ReconstructedDataParallel
from NeRN.tasks.resnet18 import ResNet18, ReconstructedResNet18
from NeRN.tasks.resnet20 import ResNet20, ReconstructedResNet20
from NeRN.tasks.resnet56 import ResNet56, ReconstructedResNet56
from NeRN.options import Config


def load_original_model(cfg, device):
    model_kwargs_path = cfg.original_model_path.replace('pt', 'json')
    if os.path.exists(model_kwargs_path):
        with open(model_kwargs_path) as f:
            model_kwargs = json.load(f)
    else:
        model_kwargs = dict()
    original_model, reconstructed_model = ModelFactory.get(cfg, device, **model_kwargs)
    return original_model, reconstructed_model


class ModelFactory:
    models = {
        "ResNet18": (ResNet18, ReconstructedResNet18),
        "ResNet20": (ResNet20, ReconstructedResNet20),
        "ResNet56": (ResNet56, ReconstructedResNet56)
    }

    @staticmethod
    def get(cfg: Config, device: torch.device, **kwargs):
        if cfg.task.original_model_name not in ModelFactory.models:
            raise ValueError("Unsupported original model name")

        model = ModelFactory.models[cfg.task.original_model_name][0](**kwargs).to(device)
        model.load(cfg.original_model_path, device)
        reconstructed_model = ModelFactory.models[cfg.task.original_model_name][1](model, cfg, device=device,
                                                                                   sampling_mode=cfg.nern.sampling_mode).to(
            device)
        if cfg.num_gpus > 1 and not cfg.no_cuda:
            model = OriginalDataParallel(model, device_ids=list(range(cfg.num_gpus)))
            reconstructed_model = ReconstructedDataParallel(reconstructed_model, device_ids=list(range(cfg.num_gpus)))

        return model, reconstructed_model
