from abc import ABC, abstractmethod
from typing import Callable, Tuple

import torch
from torch.autograd.grad_mode import F
from torch.utils.data import DataLoader

from HAND.models.model import ReconstructedModel


class EvalFunction:
    def eval(self, reconstructed_model: ReconstructedModel, dataloader: DataLoader):
        test_loss, correct = self._eval_model(reconstructed_model, dataloader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(dataloader.dataset),
            100. * correct / len(dataloader.dataset)))

    @staticmethod
    def _eval_model(reconstructed_model: ReconstructedModel, dataloader: DataLoader) -> Tuple[float, float]:
        reconstructed_model.eval()
        device = next(reconstructed_model.parameters()).device  # assuming all parameters are on the same device.
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                output = reconstructed_model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(dataloader.dataset)
        return test_loss, correct
