from typing import List

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from torch.optim.adamw import AdamW
from torch.optim.adadelta import Adadelta

from .ranger import Ranger
from NeRN.options import Config


class OptimizerFactory:
    @staticmethod
    def get(parameters: List[torch.Tensor], cfg: Config) -> Optimizer:
        if cfg.optim.optimizer_type == "sgd":
            return OptimizerFactory._init_sgd(parameters, cfg)
        elif cfg.optim.optimizer_type == "adamw":
            return OptimizerFactory._init_adamw(parameters, cfg)
        elif cfg.optim.optimizer_type == "adadelta":
            return OptimizerFactory._init_adadelta(parameters, cfg)
        elif cfg.optim.optimizer_type == "ranger":
            return OptimizerFactory._init_ranger(parameters, cfg)
        else:
            raise ValueError("Unknown Optimizer Type")

    @staticmethod
    def _init_sgd(parameters: List[torch.Tensor], cfg: Config):
        return SGD(parameters, lr=cfg.optim.lr, momentum=cfg.optim.momentum, weight_decay=cfg.optim.weight_decay,
                   nesterov=True)

    @staticmethod
    def _init_adamw(parameters: List[torch.Tensor], cfg: Config):
        return AdamW(parameters, lr=cfg.optim.lr, betas=cfg.optim.betas, weight_decay=cfg.optim.weight_decay)

    @staticmethod
    def _init_adadelta(parameters: List[torch.Tensor], cfg: Config):
        return Adadelta(parameters, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    @staticmethod
    def _init_ranger(parameters: List[torch.Tensor], cfg: Config):
        return Ranger(parameters, lr=cfg.optim.lr, betas=cfg.optim.betas, weight_decay=cfg.optim.weight_decay,
                      use_gc=cfg.optim.ranger_use_gc, gc_conv_only=False)
