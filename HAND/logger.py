import os
from datetime import datetime


def create_experiment_dir(log_dir: str, exp_name: str):
    # Code to acquire resource, e.g.:
    date_time = datetime.now().strftime("%d_%m_%Y_%H%M%S")
    dir_path = os.path.join(log_dir, f"{exp_name}_{date_time}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path
