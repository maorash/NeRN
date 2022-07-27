import pyrallis
from dataclasses import dataclass, field


@dataclass
class EmbeddingsConfig:
    # Number of indices to encode
    num_idxs: int = field(default=3)
    # Encoding levels
    enc_levels: int = field(default=20)
    # Base num
    base: float = field(default=1.25)
    # Embedding fusion mode
    fusion_mode: str = field(default='concat')
    # Indices normalization mode, if None don't normalize indices (None/local/global)
    normalization_mode: str = field(default='local')


@dataclass
class HANDConfig:
    # Initialization method (fmod/default/checkpoint)
    init: str = field(default="fmod")
    # Path for checkpoint (weights initialization)
    checkpoint_path: str = field(default=None)
    # Predictor type
    method: str = field(default='3x3')
    # Normalization layer
    norm_layer: str = field(default='bn')
    # Activation layer
    act_layer: str = field(default='relu')
    # Number of linear blocks
    num_blocks: int = field(default=3)
    # Number of linear blocks
    hidden_layer_size: int = field(default=30)
    # Positional embeddings config
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    # Batch size for weight prediction (number of weights to predict in a batch)
    weights_batch_size: int = field(default=2 ** 16)
    # Whether or not to compute gradients on the entire predicted network, or for a random batch of predicted weights
    weights_batch_gradients: bool = field(default=True)
    # Task loss weight
    task_loss_weight: float = field(default=1)
    # Reconstruction loss weight
    reconstruction_loss_weight: float = field(default=1)
    # Feature maps distillation loss weight
    attention_loss_weight: float = field(default=1)
    # Output distillation loss weight
    distillation_loss_weight: float = field(default=1)
    # Task loss type, should be a member of `torch.nn.functional`, default is `nll_loss`
    task_loss_type: str = field(default='CELoss')
    # Reconstruction loss type, should be a member of `torch.nn`, default is `MSELoss`
    reconstruction_loss_type: str = field(default='L2')
    # Feature maps distillation loss type, should be a member of `torch.nn`, default is `MSELoss`
    attention_loss_type: str = field(default='L2')
    # Output distillation loss type, should be a member of `torch.nn`, default is `KLDivLoss`
    distillation_loss_type: str = field(default='KLDivLoss')


@dataclass
class LogConfig:
    # Task name
    exp_name: str = field(default='HAND_Train')
    # How often to log metrics
    log_interval: int = field(default=20)
    # Log dir
    log_dir: str = field(default='outputs')
    # Disable logging for faster development
    disable_logging: bool = field(default=False)
    # Log gradient norms and weight norms for all layers
    verbose: bool = field(default=False)


@dataclass
class TaskConfig:
    # Task name
    task_name: str = field(default='mnist')
    # Original network type
    original_model_name: str = field(default='SimpleNet')


@dataclass
class OptimizationConfig:
    # Learning rate
    lr: float = field(default=1e-4)
    # Learning rate type
    lr_scheduler_type: str = field(default='cosine')
    # Minimum learning rate for scheduler
    min_lr: float = field(default=1e-6)
    # Gamma factor for exponential LR decay (ExponentialLR)
    # Set 0 to automatically compute the factor for achieving min_lr after all training iterations
    gamma: float = field(default=0)
    # Beta for adam. default=0.5
    betas: tuple = field(default=(0.5, 0.999))
    # Momentum for SGD optimizer
    momentum: float = field(default=0.9)
    # Weight decay for optimizer
    weight_decay: float = field(default=1e-3)
    # Optimizer to use, should be a member of `torch.optim`, default is `AdamW`
    optimizer: str = field(default='adamw')
    # Apply gradient normalization during training (set None to skip the norm clipping)
    max_gradient_norm: float = field(default=None)
    # Epochs to decay learning rate by 10
    # lr_steps: float = field(default=-1)


@dataclass
class TrainConfig:
    # Path to the original model file
    original_model_path: str = field(default='trained_models/original_tasks/mnist/mnist_cnn.pt')
    # HAND config
    hand: HANDConfig = field(default_factory=HANDConfig)
    # Number of data loading workers
    # workers: int = field(default=4)
    # Input batch size
    batch_size: int = field(default=256)
    # Resuming start_epoch from checkpoint
    # not_resume_epoch: bool = field(default=True)
    # Number of epochs to train for
    epochs: int = field(default=250)
    # Epoch cycles for trainings
    # cycles: int = field(default=1)
    # Number of warmup epochs
    # warmup: int = field(default=5)
    # How often to test the reconstructed model on the original task
    eval_epochs_interval: int = field(default=1)
    # How often to save the learned model
    save_epoch_interval: int = field(default=10)
    # Use cpu instead of cuda
    no_cuda: bool = field(default=False)
    # Learn the fully connected layer of the reconstructed model
    learn_fc_layer: bool = field(default=False)
    # Log config
    logging: LogConfig = field(default_factory=LogConfig)
    # Task config
    task: TaskConfig = field(default_factory=TaskConfig)
    # Optimization config
    optim: OptimizationConfig = field(default_factory=OptimizationConfig)
    # Num epochs to run with reconstruction loss only at the beginning of training
    loss_warmup_epochs: int = field(default=1)


@pyrallis.wrap()
def get_train_config(cfg: TrainConfig):
    return cfg
