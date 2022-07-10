import os
from datetime import datetime

from clearml import Task


def create_experiment_dir(log_dir: str, exp_name: str):
    # Code to acquire resource, e.g.:
    date_time = datetime.now().strftime("%d_%m_%Y_%H%M%S")
    dir_path = os.path.join(log_dir, f"{exp_name}_{date_time}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def initialize_clearml_task(task_name: str) -> Task:
    return Task.init(project_name='HAND_compression', task_name=task_name, deferred_init=True)
