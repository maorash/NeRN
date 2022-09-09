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
    # Is predictor wraped with data parallel
    is_data_parallel: bool = field(default=True)
    # pruning method: 'reconstruction', 'magnitude' or 'random'
    pruning_method: str = field(default='reconstruction')
    # reconstruction_error_metric: 'absolute', 'relative' or 'relative_squared'
    reconstruction_error_metric: str = field(default='relative')

@pyrallis.wrap()
def get_prune_config(cfg: PruneConfig):
    return cfg
