from abc import ABC

from torch import nn


class LossBase(nn.Module, ABC):
    # TODO: during development this API was broken,
    #  and different losses requires different parameters in the forward function. Fix this.
    pass
    # @abstractmethod
    # def forward(self,
    #             reconstructed_model: ReconstructedModel,
    #             original_model: OriginalModel,
    #             batch: Optional[torch.Tensor]) \
    #         -> torch.Tensor:
    #     raise NotImplementedError()

