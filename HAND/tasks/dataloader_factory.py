from HAND.tasks.mnist.mnist_train_main import get_dataloaders as mnist_dataloaders
from HAND.tasks.cifar10.cifar10_train_main import get_dataloaders as cifar10_dataloaders
from HAND.tasks.cifar10.akamaster_trainer import get_dataloaders as akamaster_dataloaders


class DataloaderFactory:
    task_dataloaders = {
        "mnist": mnist_dataloaders,
        "cifar10": cifar10_dataloaders,
        "akamaster": akamaster_dataloaders
    }

    @staticmethod
    def get(task_name, **kwargs):
        try:
            dataloaders = DataloaderFactory.task_dataloaders[task_name](test_kwargs=kwargs, train_kwargs=kwargs)
        except KeyError:
            raise ValueError("Unsupported task")
        return dataloaders
