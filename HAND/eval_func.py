from abc import ABC, abstractmethod
from typing import Callable, Tuple

from torch.utils.data import DataLoader

from HAND.models.model import ReconstructedModel


class EvalFunction:
    @staticmethod
    @abstractmethod
    def _eval_model(reconstructed_model: ReconstructedModel, dataloader: DataLoader) -> Tuple[float, float]:
        """

        Args:
            reconstructed_model:
            dataloader:

        Returns:
            test_loss, correct
        """
        pass

    def eval(self, reconstructed_model: ReconstructedModel, dataloader: DataLoader):
        test_loss, correct = self._eval_model(reconstructed_model, dataloader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(dataloader.dataset),
            100. * correct / len(dataloader.dataset)))