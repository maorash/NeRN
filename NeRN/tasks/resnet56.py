from typing import List, Tuple

from NeRN.models.model import ReconstructedModel
from NeRN.options import Config
from NeRN.tasks.cifar10.cifar_resnet import ResNet, BasicBlockA, BasicBlockB


class ResNet56(ResNet):
    def __init__(self, basic_block_option='A', **kwargs):
        self.basic_block_option = basic_block_option
        basic_block = BasicBlockA if self.basic_block_option == 'A' else BasicBlockB
        super(ResNet56, self).__init__(basic_block, [9, 9, 9], **kwargs)


class ReconstructedResNet56(ReconstructedModel):
    def __init__(self, original_model: ResNet56, train_cfg: Config, device: str, sampling_mode: str = None):
        super(ReconstructedResNet56, self).__init__(original_model, train_cfg, device, sampling_mode)

    def __str__(self):
        return f"{type(self).__name__}_{self.original_model.basic_block_option}"
