from typing import Optional, List, Any

import pyrallis
from dataclasses import dataclass, field
from HAND.options import TrainConfig


@dataclass
class PruneConfig:
    # Path to the HAND predictor file
    predictor_path: str = field(default='trained_models/original_tasks/mnist/mnist_cnn.pt')
    # percentile of weights to prune from the model
    pruning_factor: float = field(default=0.1)
    # Train config
    train_cfg: TrainConfig = field(default_factory=TrainConfig)

@pyrallis.wrap()
def get_prune_config(cfg: PruneConfig):
    return cfg
