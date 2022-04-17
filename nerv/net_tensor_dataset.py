import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from typing import List

tensor_list = List[torch.TensorType]


class NetTensorDataset(Dataset):
    def __init__(self, model_pth: str, layers_to_learn_prefix: str = 'convs'):
        self.model_path = model_pth
        self.layers_to_learn = layers_to_learn_prefix
        self.tensors_to_learn = self._get_tensors_to_learn(self.model_path, self.layers_to_learn)
        self.indexed_tensors = list(enumerate(self.tensors_to_learn))

    @staticmethod
    def _get_tensors_to_learn(model_path: str, layers_to_learn: str) -> tensor_list:
        tensors = []
        loaded_model = torch.load(model_path)
        for layer_name, tensor in loaded_model.items():
            if layer_name.startswith(layers_to_learn) and 'weight' in layer_name:
                tensors.append(tensor)
        return tensors

    def __getitem__(self, index) -> T_co:
        return self.indexed_tensors[index], torch.tensor(index)

    def __len__(self):
        return len(self.indexed_tensors)
