import torch
from torch import nn
from typing import List, Tuple

from abc import abstractmethod


class OriginalModel(nn.Module):
    @abstractmethod
    def get_feature_maps(self, batch: torch.TensorType) -> List[torch.TensorType]:
        pass

    @abstractmethod
    def get_learnable_weights(self):
        pass


class ReconstructedModel(OriginalModel):
    def __init__(self, original_model: OriginalModel):
        super().__init__()
        self.original_model = original_model

    @abstractmethod
    def get_positional_embeddings(self) -> Tuple[List[Tuple], List[torch.TensorType]]:
        pass

    @abstractmethod
    def update_weights(self, index: Tuple, weight: torch.TensorType):
        pass

    def get_feature_maps(self, batch: torch.TensorType) -> List[torch.TensorType]:
        return self.original_model.get_feature_maps(batch)

    def get_learnable_weights(self):
        return self.original_model.get_learnable_weights()

    def forward(self, input):
        return self.original_model(input)
