import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from HAND.tasks.mnist.mnist_train_main import get_dataloaders as mnist_dataloaders
from HAND.tasks.cifar10.cifar10_basic_train import get_dataloaders as cifar10_dataloaders
from HAND.tasks.cifar10.cifar10_akamaster_train import get_dataloaders as cifar10v2_dataloaders


class RandomDataset(Dataset):
    def __init__(self, input_shape, num_samples=5e4):
        self.input_shape = input_shape
        self.num_samples = int(num_samples)

    def __getitem__(self, item):
        return torch.randn(self.input_shape), -1  # no labels for random inputs

    def __len__(self):
        return self.num_samples


class DataloaderFactory:
    tasks_data = {
        "mnist": {
            "loader": mnist_dataloaders,
            "input_shape": (1, 28, 28),
        },
        "cifar10": {
            "loader": cifar10_dataloaders,
            "input_shape": (3, 32, 32)
        },

        "cifar10v2": {
            "loader": cifar10v2_dataloaders,
            "input_shape": (3, 32, 32)
        }
    }

    @staticmethod
    def get(task_name, use_random_inputs, **kwargs):
        try:
            task_data = DataloaderFactory.tasks_data[task_name]
            dataloaders = task_data["loader"](test_kwargs=kwargs, train_kwargs=kwargs)
            if use_random_inputs is True:
                random_dataset = RandomDataset(task_data["input_shape"])
                random_dataloder = DataLoader(random_dataset, **kwargs)
                new_dataloaders = random_dataloder, dataloaders[1]
                dataloaders = new_dataloaders
        except KeyError:
            raise ValueError("Unsupported task")
        return dataloaders
