import torch
from torch import nn
from typing import List, Tuple

from abc import ABC, abstractmethod


class ModelBase(ABC, nn.Module):
    @abstractmethod
    def get_weights(self):
        raise NotImplementedError()

    @abstractmethod
    def get_feature_maps(self, batch: torch.TensorType) -> List[torch.TensorType]:
        raise NotImplementedError()


class ReconstructedModel(ModelBase):
    @abstractmethod
    def get_feature_maps(self, batch: torch.TensorType) -> List[torch.TensorType]:
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def get_reconstructed_model(self) -> nn.Module:
        pass

    @abstractmethod
    def get_positional_embeddings(self) -> Tuple[List[Tuple], List[torch.TensorType]]:
        pass

    @abstractmethod
    def update_weights(self, index: Tuple, weight: torch.TensorType):
        pass


class OriginalModel(ModelBase):
    @abstractmethod
    def get_feature_maps(self, batch: torch.TensorType) -> List[torch.TensorType]:
        pass

    @abstractmethod
    def get_weights(self):
        pass
