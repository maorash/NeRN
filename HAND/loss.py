from abc import ABC, abstractmethod
from typing import Tuple

import torch

from torch import nn

from HAND.model import ReconstructedModel, OriginalModel


class LossBase(nn.Module, ABC):
    @staticmethod
    @abstractmethod
    def forward(reconstructed_model: ReconstructedModel, original_model: OriginalModel):
        raise NotImplementedError()


class ReconstructionLoss(LossBase):
    @staticmethod
    def forward(reconstructed_model: ReconstructedModel, original_model: OriginalModel):
        raise NotImplementedError()


class DistillationLoss(LossBase):
    @staticmethod
    def forward(reconstructed_model: ReconstructedModel, original_model: OriginalModel):
        raise NotImplementedError()
