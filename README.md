# NeRN 
### LEARNING NEURAL REPRESENTATIONS FOR NEURAL NETWORKS

Link to paper: https://arxiv.org/abs/2212.13554

This is the official framework for "NeRN - Learning Neural Representations for Neural Networks". 
This framework can be used to recreate the experiments detailed in the paper, or be extended to support additional networks and tasks.

## 💿 Installation

The requirements for NeRN are listed in `requirements.txt`.
To create a new _conda_ environment with the required packages, run the following:
```sh
conda create --name nern python=3.7
conda activate nern
pip install -r requirements.txt
```
## 🏃 Training a NeRN

#### Requirements
- A working python environment.
- Pretrained CNN weights
    - Current supported architectures are ResNet18, ResNet20 & ResNet56.
    - Current supported benchmarks are CIFAR-10, CIFAR-100 & ImageNet.
    - Below we list instructions for extending the framework to new networks and tasks.
- A _pyrallis_ configuration file (_.yaml_).
- (Hopefully) A GPU.

#### Example Configuration File
```
logging:
  exp_name: cifar10_resnet20_smoothpes_permute_crossfilter_180  # Will be used for logging and checkpoints
  log_dir: outputs/resnet20/cifar10
original_model_path: trained_models/original_tasks/cifar10/resnet20_0_B.pt  # Path to original pretrained network
nern:
  num_blocks: 5
  hidden_layer_size: 180
  weights_batch_method: random_weighted_batch  # This is the magnitude-oriented batching
  embeddings:
    enc_levels: 40  # For three indices we get a positional embedding of size (40 * 3) * 2 = 240
    base: 0.76  # Smooth positional embeddings
    permutations:
      permute_mode: crossfilter
  weights_batch_size: 4096
  distillation_loss_type: StableKLDivLoss
task:
  task_name: cifar10
  original_model_name: ResNet20
optim:
  optimizer_type: ranger
  lr: 5e-3
  ranger_use_gc: False
epochs: 350  # Either set epochs or iterations
num_gpus: 1
num_workers: 12
```

#### Instructions
The NeRN framework supports both ClearML and TensorBoard for logging, and it's up to you to decide which one you prefer. The default is ClearML, but can be changed by adding the following to the configuration file:
```
logging:
    use_tensorboard: True
```
Finally, run:
```
python train.py --config_path <PATH_TO_CONFIG_YAML_FILE>
```
It is possible to override some fields in the configuration file in the execution line. For example:
```
python train.py --logging.use_tensorboard True --epochs 400 --nern.weights_batch_method all --config_path <PATH_TO_CONFIG_YAML_FILE>
```

## 📊 Evaluating a NeRN
Create a copy of the configure used to train NeRN, and add the following fields
```
nern:
  init: checkpoint
  checkpoint_path: <PATH_TO_NERN_CHECKPOINT>
```
These will initialize the NeRN using pretrained weights (by the way, this can also be used for finetuning/transfer learning)
Then, run:
```
python evaluate.py --config_path <PATH_TO_EVAL_CONFIG_YAML_FILE>
```

## 🔧 Extending the Framework
### Adding new benchmarks
Under the package `tasks`, create function implementation to retrieve the _test_ and _train_ datasets. This function will recieve *train_kwargs*, *test_kwargs* and an optional *use_workers* flag. The function will return the training dataset and testing dataset, for example:
```python
def get_dataloaders(train_kwargs, test_kwargs, use_workers=True, **kwargs):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=train_kwargs["batch_size"], shuffle=True,
        num_workers=train_kwargs["num_workers"] if use_workers else 0, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=test_kwargs["num_workers"] if use_workers else 0, pin_memory=True)
    return train_loader, val_loader
```
Then, under `dataloader_factory.py`, add your new benchmark in the `DataLoaderFactory` as such:
```
class DataloaderFactory:
    tasks_data = {
        ...
        "mybenchmark": {
            "loader": my_get_dataloaders,
            "input_shape": (3, 32, 32)
        },
        ...
    }
```
### Adding new networks
#### Step 1:
Create an `OriginalModel` wrapper, and implement the following methods:
```
class MyNewModel(OriginalModel):
    def __init__(self, num_classes: int = 10, **kwargs):
        super(MyNewModel, self).__init__()
        ...

    def get_feature_maps(self, batch: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Given an input batch, returns a list of feature maps from all layers of the network (before the activation function)
        pass

    def get_learnable_weights(self) -> List[torch.Tensor]:
        # Returns a list containing the weights of all the convolutional layers in the network which we wish to represent using NeRN
        pass

    def load(self, path: str, device: torch.device):
        # Loads weights from a given path to your model
        pass
```
#### Step 2:
Create a `ReconstructedModel` wrapper, and implement the `__str__` method:
```
class ReconstructedMyNewModel(ReconstructedModel):
    def __init__(self, original_model: ResNet20, train_cfg: Config, device: str,
                 sampling_mode: str = None):
        super(ReconstructedResNet20, self).__init__(original_model, train_cfg, device,  sampling_mode)

    def __str__(self):
        # Returns the name of the model in a unique way. Will be used for caching purposes
        pass
```
#### Step 3:
Add your classes to the `ModelFactory`:
```
class ModelFactory:
    models = {
        ...
        "MyNewModel": (MyNewModel, ReconstructedMyNewModel),
        ...
    }
```

## 📃 Final Note
Under `experiments`, you will find all the relevant configuration files, used to generate the results for sections `4.1`-`4.3` in the paper.
The NeRN framework is still under active development, we hope you have fun using it and welcome any feedbacks.


## Citation
If you use this code or paper for your research, please cite the following:


```
@misc{https://doi.org/10.48550/arxiv.2212.13554,
  doi = {10.48550/ARXIV.2212.13554},
  url = {https://arxiv.org/abs/2212.13554},
  author = {Ashkenazi, Maor and Rimon, Zohar and Vainshtein, Ron and Levi, Shir and Richardson, Elad and Mintz, Pinchas and Treister, Eran},
  keywords = {Machine Learning (cs.LG), Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {NeRN -- Learning Neural Representations for Neural Networks},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```