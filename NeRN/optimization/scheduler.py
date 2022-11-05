from typing import List

import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from NeRN.options import Config


class GenericScheduler:
    def __init__(self, scheduler: _LRScheduler, min_lr: float, step_interval: int):
        self.scheduler = scheduler
        self.min_lr = min_lr
        self.step_interval = step_interval

    def check_and_step(self, step: int):
        if step % self.step_interval == 0 and self.min_lr < self.get_last_lr():
            self.scheduler.step()

    def get_last_lr(self) -> float:
        return self.scheduler.get_last_lr()[-1]


class LRSchedulerFactory:
    @staticmethod
    def get(optimizer: Optimizer, cfg: Config) -> GenericScheduler:
        if cfg.optim.lr_scheduler_type == "cosine":
            return LRSchedulerFactory._init_cosine(optimizer, cfg)
        elif cfg.optim.lr_scheduler_type == "exponential":
            return LRSchedulerFactory._init_exponential(optimizer, cfg)
        else:
            raise ValueError("Unknown LR Scheduler Type")

    @staticmethod
    def _init_cosine(optimizer: Optimizer, cfg: Config):
        return GenericScheduler(CosineAnnealingLR(optimizer, T_max=cfg.num_iterations, eta_min=cfg.optim.min_lr),
                                min_lr=cfg.optim.min_lr,
                                step_interval=cfg.optim.step_interval)

    @staticmethod
    def _init_exponential(optimizer: Optimizer, cfg: Config, default_steps=100):
        # If not explicitly set, automatically compute exponential factor to achieve min_lr at final iteration
        step_interval = cfg.optim.step_interval if cfg.optim.step_interval > 0 else cfg.num_iterations // default_steps
        approx_num_steps = cfg.num_iterations / step_interval
        gamma = np.power(cfg.optim.min_lr / cfg.optim.lr, 1 / approx_num_steps) if cfg.optim.gamma == 0 else cfg.optim.gamma
        return GenericScheduler(ExponentialLR(optimizer, gamma),
                                min_lr=cfg.optim.min_lr,
                                step_interval=step_interval)
