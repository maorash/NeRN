import collections
import os
from datetime import datetime
from typing import List, Union

from clearml import Logger
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor


def create_experiment_dir(log_dir: str, exp_name: str):
    date_time = datetime.now().strftime("%d_%m_%Y_%H%M%S")
    dir_path = os.path.join(log_dir, f"checkpoints_{exp_name}_{date_time}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def log_scalar_list(scalar_list: list, title: str, series_name: str, iteration: int, logger: Union[Logger, SummaryWriter]):
    if isinstance(logger, SummaryWriter):
        for i, scalar in enumerate(scalar_list):
            logger.add_scalar(tag=f'{title}_{series_name}_{i}', scalar_value=scalar, global_step=iteration)
    else:
        for i, scalar in enumerate(scalar_list):
            logger.report_scalar(title=title, series=f'{series_name}_{i}', value=scalar, iteration=iteration)


def log_scalar_dict(scalar_dict: dict, title: str, iteration: int, logger: Union[Logger, SummaryWriter]):
    if isinstance(logger, SummaryWriter):
        for key, value in scalar_dict.items():
            logger.add_scalar(tag=f'{title}/{key}', scalar_value=value, global_step=iteration)
    else:
        for key, value in scalar_dict.items():
            logger.report_scalar(title=title, series=key, value=value, iteration=iteration)


def log_scalar(scalar: float, title: str, series: str, iteration: int, logger: Union[Logger, SummaryWriter]):
    if isinstance(logger, SummaryWriter):
        logger.add_scalar(tag=f'{title}/{series}', scalar_value=scalar, global_step=iteration)
    else:
        logger.report_scalar(title=title, series=series, value=scalar, iteration=iteration)


def compute_grad_norms(weights: List[Tensor]):
    return [weight.grad.norm() for weight in weights]


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
