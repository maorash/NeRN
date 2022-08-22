from typing import Union

import torch
from clearml import Logger
from torch.nn.functional import nll_loss
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from HAND.options import TrainConfig
from HAND.models.model import ReconstructedModel, ReconstructedDataParallel
from HAND.tasks.imagenet_helpers import validate


class EvalFunction:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

    def eval(self, reconstructed_model: Union[ReconstructedModel, ReconstructedDataParallel],
             dataloader: DataLoader,
             iteration: int,
             clearml_logger: Logger,
             suffix=""):
        if self.cfg.task.task_name == "imagenet":
            return self._logged_imagenet_eval(reconstructed_model, dataloader, iteration, clearml_logger, suffix)
        else:
            return self._basic_eval(reconstructed_model, dataloader, iteration, clearml_logger, suffix)

    def _logged_imagenet_eval(self, reconstructed_model: Union[ReconstructedModel, ReconstructedDataParallel],
                              dataloader: DataLoader,
                              iteration: int,
                              clearml_logger: Logger,
                              suffix=""):
        metrics = validate(reconstructed_model, dataloader, CrossEntropyLoss(), log_suffix=suffix)
        if clearml_logger is not None:
            for m in metrics.keys():
                clearml_logger.report_scalar(f'{m}{f"_{suffix}" if suffix else ""}', m, metrics[m], iteration)

        return metrics['top1_accuracy']

    def _basic_eval(self, reconstructed_model: Union[ReconstructedModel, ReconstructedDataParallel],
                    dataloader: DataLoader,
                    iteration: int,
                    clearml_logger: Logger,
                    suffix="") -> float:
        print(f'\n Starting eval on test set{f" - {suffix}" if suffix else "."}')
        reconstructed_model.eval()
        device = next(reconstructed_model.parameters()).device  # assuming all parameters are on the same device.

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                output = reconstructed_model(data)
                test_loss += nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(dataloader.dataset)

        accuracy = 100. * correct / len(dataloader.dataset)
        if clearml_logger is not None:
            clearml_logger.report_scalar(f'eval_loss{f"_{suffix}" if suffix else ""}', 'eval_loss', test_loss, iteration)
            clearml_logger.report_scalar(f'eval_accuracy{f"_{suffix}" if suffix else ""}', 'eval_accuracy', accuracy,
                                         iteration)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(dataloader.dataset),
            accuracy))

        return accuracy
