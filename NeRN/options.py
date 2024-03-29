from typing import Optional, List, Any

import pyrallis
from dataclasses import dataclass, field


@dataclass
class PermutationsConfig:
    # The permutation smoothing mode (none/infilter/crossfilter)
    permute_mode: str = field(default=None)
    # Number of workers to use for crossfilter permutations
    num_workers: int = field(default=3)
    # Permutations cache folder path (if empty, a default path will be chosen based on the configuration)
    path: Optional[str] = field(default=None)


@dataclass
class EmbeddingsConfig:
    # Type of positional embedding to use (basic/ffn)
    type: str = field(default='basic')
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
    # Gaussian kernel scale for ffn (fourier feature network)
    gauss_scale: List[float] = field(default_factory=lambda: [1, 0.1, 0.1])
    # Permutations config
    permutations: PermutationsConfig = field(default_factory=PermutationsConfig)


@dataclass
class NeRNConfig:
    # Initialization method (fmod/default/checkpoint)
    init: str = field(default="fmod")
    # Path for checkpoint (weights initialization)
    checkpoint_path: Optional[str] = field(default=None)
    # Predictor type
    method: str = field(default='kxk')
    # Output size for the kxk method
    output_size: int = field(default=3)
    # Normalization layer
    norm_layer: str = field(default='bn')
    # Activation layer
    act_layer: str = field(default='relu')
    # Number of linear blocks
    num_blocks: int = field(default=3)
    # Number of linear blocks
    hidden_layer_size: int = field(default=30)
    # Batch size for weight prediction (number of weights to predict in a batch)
    weights_batch_size: Optional[int] = field(default=2 ** 16)
    # Weight batching method (all/sequential_layer/random_layer/random_batch/random_batch_without_replacement)
    weights_batch_method: str = field(default='all')
    # Positional embeddings config
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    # Reconstruction loss weight
    reconstruction_loss_weight: float = field(default=1)
    # Feature maps distillation loss weight
    attention_loss_weight: float = field(default=0.00001)
    # Output distillation loss weight
    distillation_loss_weight: float = field(default=0.00001)
    # Task loss type
    task_loss_type: str = field(default='CELoss')
    # Reconstruction loss type
    reconstruction_loss_type: str = field(default='L2Loss')
    # Feature maps distillation loss type
    attention_loss_type: str = field(default='L2BatchedAttentionLoss')
    # Output distillation loss type
    distillation_loss_type: str = field(default='KLDivLoss')
    # The sampling mode for the reconstruction model (center/average/max)
    sampling_mode: str = field(default='center')


@dataclass
class LogConfig:
    # Task name
    exp_name: str = field(default='NeRN_Train')
    # How often to log metrics in iterations (batches)
    log_interval: int = field(default=20)
    # Log dir
    log_dir: str = field(default='outputs')
    # Disable logging for faster development
    disable_logging: bool = field(default=False)
    # Log gradient norms and weight norms for all layers
    verbose: bool = field(default=False)
    # Use tensorboardX for logging
    use_tensorboard: bool = field(default=False)


@dataclass
class TaskConfig:
    # Task name
    task_name: str = field(default='mnist')
    # ImageNet path
    imagenet_path: str = field(default=None)
    # Original network type
    original_model_name: str = field(default='SimpleNet')
    # Whether to use random input data
    use_random_data: bool = field(default=False)


@dataclass
class OptimizationConfig:
    # Optimizer to use, should be a member of `torch.optim`, default is `AdamW`
    optimizer_type: str = field(default='adamw')
    # Learning rate
    lr: float = field(default=1e-4)
    # Beta for adam/ranger. default=0.5
    betas: tuple = field(default=(0.5, 0.999))
    # Momentum for SGD optimizer
    momentum: float = field(default=0.9)
    # Weight decay for optimizer
    weight_decay: float = field(default=1e-3)
    # Ranger optimizer - use gradient centralization
    ranger_use_gc: bool = field(default=True)
    # Apply gradient normalization during training (set None to skip the norm clipping)
    max_gradient_norm: Optional[float] = field(default=None)
    # Apply gradient clipping during training (set None to skip the clipping)
    max_gradient: Optional[float] = field(default=None)
    # Learning rate scheduler type
    lr_scheduler_type: str = field(default='cosine')
    # Minimum learning rate for scheduler
    min_lr: float = field(default=1e-6)
    # Set 0 to automatically compute step interval. In cosine lr decay, will be enforced to 1
    step_interval: int = field(default=1)
    # Gamma factor for exponential LR decay (ExponentialLR)
    # Set 0 to automatically compute the factor for achieving min_lr after all training iterations
    gamma: float = field(default=0)


@dataclass
class Config:
    # Path to the original model file
    original_model_path: Optional[str] = field(default=None)
    # NeRN config
    nern: NeRNConfig = field(default_factory=NeRNConfig)
    # Number of data loading workers
    num_workers: int = field(default=0)
    # Number of available GPUs
    num_gpus: int = field(default=1)
    # Input batch size
    batch_size: int = field(default=256)
    # Number of epochs to train for, mutually exclusive with num_iterations
    epochs: Optional[int] = field(default=250)
    # Number of iterations to train for, mutually exclusive with epochs
    num_iterations: Optional[int] = field(default=None)
    # How often to test the reconstructed model on the original task (in epochs)
    eval_epochs_interval: Optional[int] = field(default=1)
    # How often to save the learned model (in epochs)
    save_epochs_interval: Optional[int] = field(default=400)
    # How often to test the reconstructed model on the original task, in iterations (batches)
    eval_iterations_interval: Optional[int] = field(default=None)
    # How often to save the learned model, in iterations (batches)
    save_iterations_interval: Optional[int] = field(default=None)
    # Loss window size (in iterations), used for a greedy selection for triggering evaluation
    eval_loss_window_size: int = field(default=20)
    # How often to add the loss to the window, number of iterations (batches)
    eval_loss_window_interval: int = field(default=10)
    # Number of iterations to optimize only using reconstruction loss at the beginning of training
    loss_warmup_iterations: int = field(default=10000)
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

    def __post_init__(self):
        def _assert_mutex(vals: List[Any], desc: str, min_set: int = 1, max_set: int = 1):
            num_set = len([v for v in vals if v is not None])
            if num_set < min_set or num_set > max_set:
                raise ValueError(f'Mutex options error: {desc}')

        _assert_mutex([self.epochs, self.num_iterations], 'epochs/num_iterations')
        if self.optim.lr_scheduler_type == 'cosine' and self.optim.step_interval != 1:
            print('WARNING: step_interval is enforced to 1 for cosine lr scheduler')


@pyrallis.wrap()
def get_train_config(cfg: Config):
    return cfg
