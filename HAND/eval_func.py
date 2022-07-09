from typing import Tuple
import torch
from torch.nn.functional import nll_loss
from torch.utils.data import DataLoader

from HAND.logger import get_clearml_logger
from HAND.models.model import ReconstructedModel


class EvalFunction:
    def eval(self, reconstructed_model: ReconstructedModel, dataloader: DataLoader, epoch: int) -> float:
        print('\n Starting eval on test set.')
        logger = get_clearml_logger()
        test_loss, correct = self._eval_model(reconstructed_model, dataloader)
        accuracy = 100. * correct / len(dataloader.dataset)
        logger.report_scalar('eval_loss', 'eval_loss', test_loss, epoch)
        logger.report_scalar('eval_accuracy', 'eval_accuracy', accuracy, epoch)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(dataloader.dataset),
            accuracy))
        return accuracy

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
                test_loss += nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(dataloader.dataset)
        return test_loss, correct
