from typing import List

import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from HAND.options import TrainConfig


class GenericScheduler:
    def __init__(self, scheduler: _LRScheduler, step_on_epoch=True, step_on_batch=False):
        self.scheduler = scheduler
        self._step_on_epoch = step_on_epoch
        self._step_on_batch = step_on_batch

    def step_epoch(self):
        if self._step_on_epoch:
            self.scheduler.step()

    def step_batch(self):
        if self._step_on_batch:
            self.scheduler.step()

    def get_last_lr(self) -> List[float]:
        return self.scheduler.get_last_lr()


class LRSchedulerFactory:
    @staticmethod
    def get(optimizer: Optimizer, num_iterations: int, cfg: TrainConfig) -> GenericScheduler:
        if cfg.optim.lr_scheduler_type == "cosine":
            return LRSchedulerFactory._init_cosine(optimizer, num_iterations, cfg)
        elif cfg.optim.lr_scheduler_type == "exponential":
            return LRSchedulerFactory._init_exponential(optimizer, cfg)
        else:
            raise ValueError("Unknown LR Scheduler Type")

    @staticmethod
    def _init_cosine(optimizer: Optimizer, num_iterations: int, cfg: TrainConfig):
        return GenericScheduler(CosineAnnealingLR(optimizer,
                                                  T_max=(cfg.epochs * num_iterations),
                                                  eta_min=cfg.optim.min_lr),
                                step_on_epoch=False,
                                step_on_batch=True)

    @staticmethod
    def _init_exponential(optimizer: Optimizer, cfg: TrainConfig):
        # If not explicitly set, automatically compute exponential factor to achieve min_lr at final iteration
        gamma = np.power(cfg.optim.min_lr / cfg.optim.lr, 1 / cfg.epochs) if cfg.optim.gamma == 0 else cfg.optim.gamma
        return GenericScheduler(ExponentialLR(optimizer, gamma),
                                step_on_epoch=True,
                                step_on_batch=False)
