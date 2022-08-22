import torch
from torch import nn

from HAND.options import HANDConfig, TaskConfig
from HAND.loss.loss import LossBase


class TaskLossBase(LossBase):
    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                **kwargs) -> torch.Tensor:
        raise NotImplementedError()


class NLLTaskLoss(TaskLossBase):
    def __init__(self):
        super().__init__()

    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                **kwargs) -> torch.Tensor:
        return nn.functional.nll_loss(prediction, target)


class CELoss(TaskLossBase):
    def __init__(self):
        super().__init__()

    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                **kwargs) -> torch.Tensor:
        return nn.CrossEntropyLoss()(prediction, target)


class StableCELoss(TaskLossBase):
    def __init__(self):
        super().__init__()

    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                **kwargs) -> torch.Tensor:
        prediction_log_softmax = torch.log(nn.Softmax(dim=1)(prediction)+1e-3)
        return nn.NLLLoss()(prediction_log_softmax, target)


class NoLoss(TaskLossBase):
    def __init__(self):
        super().__init__()

    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                **kwargs) -> torch.Tensor:
        return torch.tensor(0.).to(prediction.device)


class TaskLossFactory:
    losses = {
        "NLLLoss": NLLTaskLoss,
        "CELoss": CELoss,
        "StableCELoss": StableCELoss,
        "NoLoss": NoLoss
    }

    @staticmethod
    def get(hand_cfg: HANDConfig, task_cfg: TaskConfig) -> TaskLossBase:
        try:
            if task_cfg.use_random_data:
                return TaskLossFactory.losses["NoLoss"]()
            return TaskLossFactory.losses[hand_cfg.task_loss_type]()
        except KeyError:
            raise ValueError("Unknown Task Loss Type")
