import os
from datetime import datetime
from typing import List

from clearml import Logger
from torch import Tensor


def create_experiment_dir(log_dir: str, exp_name: str):
    date_time = datetime.now().strftime("%d_%m_%Y_%H%M%S")
    dir_path = os.path.join(log_dir, f"{exp_name}_{date_time}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def log_scalar_list(scalar_list: list, title: str, series_name: str, iteration: int, logger: Logger):
    for i, scalar in enumerate(scalar_list):
        logger.report_scalar(title=title, series=f'{series_name}_{i}', value=scalar, iteration=iteration)


def log_scalar_dict(scalar_dict: dict, title: str, iteration: int, logger: Logger):
    for key, value in scalar_dict.items():
        logger.report_scalar(title=title, series=key, value=value, iteration=iteration)


def compute_grad_norms(weights: List[Tensor]):
    return [weight.grad.norm() for weight in weights]
