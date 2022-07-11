import pyrallis
from dataclasses import dataclass, field


@dataclass
class EmbeddingsConfig:
    # Base value/embed length for position encoding
    embed: str = field(default='1.25_80')


@dataclass
class HANDConfig:
    # Predictor type
    method: str = field(default='3x3')
    # Normalization layer
    norm_layer: str = field(default='bn')
    # Activation layer
    act_layer: str = field(default='ReLU')
    # Number of linear blocks
    num_blocks: int = field(default=3)
    # Task loss weight (distillation weight will be 1 - reconstruction_loss_weight)
    task_loss_weight: float = field(default=0.25)
    # Reconstruction loss weight (distillation weight will be 1 - reconstruction_loss_weight)
    reconstruction_loss_weight: float = field(default=0.25)
    # Feature maps distillation loss weight (distillation weight will be 1 - reconstruction_loss_weight)
    feature_maps_distillation_loss_weight: float = field(default=0.25)
    # Output distillation loss weight (distillation weight will be 1 - reconstruction_loss_weight)
    output_distillation_loss_weight: float = field(default=0.25)
    # Task loss type, should be a member of `torch.nn.functional`, default is `nll_loss`
    task_loss_type: str = field(default='nll_loss')
    # Reconstruction loss type, should be a member of `torch.nn`, default is `MSELoss`
    reconstruction_loss_type: str = field(default='MSELoss')
    # Feature maps distillation loss type, should be a member of `torch.nn`, default is `MSELoss`
    feature_maps_distillation_loss_type: str = field(default='MSELoss')
    # Output distillation loss type, should be a member of `torch.nn`, default is `KLDivLoss`
    output_distillation_loss_type: str = field(default='KLDivLoss')


@dataclass
class LogConfig:
    # Task name
    task_name: str = field(default='HAND_Train')
    # How often to log metrics
    log_interval: int = field(default=20)
    # Log dir
    log_dir: str = field(default='outputs')
    # Disable logging for faster development
    disable_logging: bool = field(default=False)


@dataclass
class TrainConfig:
    # The experiment name
    exp_name: str = field(default='default_exp')
    # Embeddings config
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    # Path to the original model file
    original_model_path: str = field(default='trained_models/original_tasks/mnist/mnist_cnn.pt')
    # HAND config
    hand: HANDConfig = field(default_factory=HANDConfig)
    # Number of data loading workers
    workers: int = field(default=4)
    # Input batch size
    batch_size: int = field(default=100)
    # Resuming start_epoch from checkpoint
    not_resume_epoch: bool = field(default=True)
    # Number of epochs to train for
    epochs: int = field(default=100000)
    # Epoch cycles for trainings
    cycles: int = field(default=1)
    # Number of warmup epochs
    warmup: int = field(default=5)
    # Learning rate
    lr: float = field(default=0.00001)
    # Learning rate type
    lr_type: str = field(default='cosine')
    # Epochs to decay learning rate by 10
    lr_steps: float = field(default=-1)
    # Beta for adam. default=0.5
    betas: tuple = field(default=(0.5, 0.999))
    # Loss weight
    lw: float = field(default=1.0)
    # Using sigmoid for output prediction
    sigmoid: bool = field(default=True)
    # Optimizer to use, should be a member of `torch.optim`, default is `AdamW`
    optimizer: str = field(default='AdamW')
    # How often to test the reconstrcuted model on the original task
    eval_epochs_interval: int = field(default=1)
    # How often to save the learned model
    save_epoch_interval: int = field(default=1000)
    # Use cpu instead of cuda
    no_cuda: bool = field(default=False)
    # Log config
    logging: LogConfig = field(default_factory=LogConfig)


@pyrallis.wrap()
def get_train_config(cfg: TrainConfig):
    return cfg
