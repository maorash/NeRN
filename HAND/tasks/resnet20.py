from typing import List, Tuple

from HAND.models.model import ReconstructedModel
from HAND.options import EmbeddingsConfig
from HAND.tasks.resnet import ResNet, BasicBlockA, BasicBlockB
from HAND.models.model import OriginalModel

class ResNet20(ResNet):
    def __init__(self, basic_block_option='A', **kwargs):
        basic_block = BasicBlockA if basic_block_option == 'A' else BasicBlockB
        super(ResNet20, self).__init__(basic_block, [3, 3, 3])


class ReconstructedResNet20(ReconstructedModel):
    def __init__(self, original_model: ResNet20, embeddings_cfg: EmbeddingsConfig, sampling_mode: str = None):
        super(ReconstructedResNet20, self).__init__(original_model, embeddings_cfg, sampling_mode)
        self.indices = self._get_tensor_indices()
        self.positional_embeddings = self._calculate_position_embeddings()

    def _get_tensor_indices(self) -> List[List[Tuple]]:
        indices = []

        for layer_idx in range(0, len(self.original_model.get_learnable_weights())):
            curr_layer_indices = []
            for filter_idx in range(self.original_model.num_hidden[layer_idx][0]):
                for channel_idx in range(self.original_model.num_hidden[layer_idx][1]):
                    curr_layer_indices.append((layer_idx, filter_idx, channel_idx))
            indices.append(curr_layer_indices)

        return indices
