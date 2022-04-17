import pyrallis
from dataclasses import dataclass, field


@dataclass
class EmbeddingsConfig:
    # Base value/embed length for position encoding
    embed: str = field(default='1.25_80')


@dataclass
class HANDConfig:
    # Prediction method
    method: str = field(default='basic')
    # Normalization layer
    norm_layer: str = field(default='bn')
    # Activation layer
    act_layer: str = field(default='relu')
    # Number of linear blocks
    num_blocks: int = field(default=3)
    # Reconstruction loss factor (distillation factor will be 1 - reconstruction_factor)
    reconstruction_factor: float = field(default=0.5)


@dataclass
class TrainConfig:
    # The experiment name
    exp_name: str = field(default='default_exp')
    # Embeddings config
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    # HAND config
    hand: HANDConfig = field(default_factory=HANDConfig)
    # Number of data loading workers
    workers: int = field(default=4)
    # Input batch size
    batch_size: int = field(default=1)
    # Resuming start_epoch from checkpoint
    not_resume_epoch: bool = field(default=True)
    # Number of epochs to train for
    epochs: int = field(default=150)
    # Epoch cycles for training
    cycles: int = field(default=1)
    # Number of warmup epochs
    warmup: int = field(default=5)
    # Learning rate
    lr: float = field(default=0.001)
    # Learning rate type
    lr_type: str = field(default='cosine')
    # Epochs to decay learning rate by 10
    lr_steps: float = field(default=-1)
    # Beta for adam. default=0.5
    beta: float = field(default=0.5)
    # Loss weight
    lw: float = field(default=1.0)
    # Using sigmoid for output prediction
    sigmoid: bool = field(default=True)


@pyrallis.wrap()
def get_train_config(cfg: TrainConfig):
    return cfg

