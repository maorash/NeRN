import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from NeRN.tasks.cifar10.cifar10_akamaster_train import get_dataloaders as cifar10_dataloaders
from NeRN.tasks.cifar10.cifar10_akamaster_train import get_cifar100_dataloaders as cifar100_dataloaders
from NeRN.tasks.imagenet_helpers import get_dataloaders as imagenet_timm_dataloaders
from NeRN.options import TaskConfig


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
        "cifar10": {
            "loader": cifar10_dataloaders,
            "input_shape": (3, 32, 32)
        },
        "cifar100": {
            "loader": cifar100_dataloaders,
            "input_shape": (3, 32, 32)
        },
        "imagenet": {
            "loader": imagenet_timm_dataloaders,
            "input_shape": (3, 224, 224)
        }
    }

    @staticmethod
    def get(task_cfg: TaskConfig, **kwargs):
        try:
            task_data = DataloaderFactory.tasks_data[task_cfg.task_name]
            dataloaders = task_data["loader"](test_kwargs=kwargs, train_kwargs=kwargs, task_cfg=task_cfg)
            if task_cfg.use_random_data is True:
                random_dataset = RandomDataset(task_data["input_shape"])
                random_dataloder = DataLoader(random_dataset, **kwargs)
                new_dataloaders = random_dataloder, dataloaders[1]
                dataloaders = new_dataloaders
        except KeyError:
            raise ValueError("Unsupported task")
        return dataloaders
