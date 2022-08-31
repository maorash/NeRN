from typing import List, Tuple

from HAND.models.model import ReconstructedModel
from HAND.options import EmbeddingsConfig
from HAND.tasks.cifar_resnet import ResNet, BasicBlockA, BasicBlockB


class ResNet20(ResNet):
    def __init__(self, basic_block_option='A', **kwargs):
        self.basic_block_option = basic_block_option
        basic_block = BasicBlockA if self.basic_block_option == 'A' else BasicBlockB
        super(ResNet20, self).__init__(basic_block, [3, 3, 3])


class ReconstructedResNet20(ReconstructedModel):
    def __init__(self, original_model: ResNet20, embeddings_cfg: EmbeddingsConfig,
                 sampling_mode: str = None):
        super(ReconstructedResNet20, self).__init__(original_model, embeddings_cfg, sampling_mode)

    def _get_tensor_indices(self) -> List[List[Tuple]]:
        indices = []
        normalize_indices = []

        max_index = max([max(weights_shape) for weights_shape in self.get_learnable_weights_shapes()])

        num_layers = len(self.original_model.get_learnable_weights())
        for layer_idx in range(0, num_layers):
            curr_layer_indices = []
            curr_normalized_layer_indices = []
            curr_num_filters = self.original_model.num_hidden[layer_idx][0]
            for filter_idx in range(curr_num_filters):
                curr_num_channels = self.original_model.num_hidden[layer_idx][1]
                for channel_idx in range(curr_num_channels):
                    curr_layer_indices.append((layer_idx, filter_idx, channel_idx))
                    if self.embeddings_cfg.normalization_mode == "None":
                        curr_normalized_layer_indices.append((layer_idx, filter_idx, channel_idx))
                    elif self.embeddings_cfg.normalization_mode == "global":
                        curr_normalized_layer_indices.append(
                            (layer_idx / max_index, filter_idx / max_index, channel_idx / max_index))
                    elif self.embeddings_cfg.normalization_mode == "local":
                        curr_normalized_layer_indices.append(
                            (layer_idx / num_layers, filter_idx / curr_num_filters, channel_idx / curr_num_channels))
                    else:
                        raise ValueError(f"Unsupported normalization mode {self.embeddings_cfg.normalization_mode}")

            indices.append(curr_layer_indices)
            normalize_indices.append(curr_normalized_layer_indices)

        self.normalized_indices = normalize_indices

        return indices

    def __str__(self):
        return f"{type(self).__name__}_{self.original_model.basic_block_option}"
