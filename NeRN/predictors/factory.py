from abc import abstractmethod
from typing import List

import torch
from torch.nn import DataParallel

from NeRN.options import Config
from NeRN.predictors.predictor import NeRNPredictorBase
from NeRN.predictors.predictor import NeRNBasicPredictor, NeRNKxKPredictor, NeRNKxKNerFPredictor, \
    NeRNKxKResidualPredictor


class NeRNPredictorFactory:
    def __init__(self, cfg: Config, input_size: int):
        self.cfg = cfg
        self.input_size = input_size

    def get_predictor(self):
        if self.cfg.nern.method == 'basic':
            predictor = NeRNBasicPredictor(self.cfg.nern, self.input_size)
        elif self.cfg.nern.method == 'kxk':
            predictor = NeRNKxKPredictor(self.cfg.nern, self.input_size)
        elif self.cfg.nern.method == 'kxk_nerf':
            predictor = NeRNKxKNerFPredictor(self.cfg.nern, self.input_size)
        elif self.cfg.nern.method == 'kxk_residual':
            predictor = NeRNKxKResidualPredictor(self.cfg.nern, self.input_size)
        else:
            raise ValueError(f'Not recognized predictor type {self.cfg.nern.method}')

        if self.cfg.num_gpus > 1:
            predictor = PredictorDataParallel(predictor, device_ids=list(range(self.cfg.num_gpus)))

        return predictor


class PredictorDataParallel(DataParallel):
    def __init__(self, module: NeRNPredictorBase, *args, **kwargs):
        DataParallel.__init__(self, module, *args, **kwargs)
        self.module = module

    def save(self, path: str):
        self.module.save(path)

    def load(self, path: str):
        self.module.load(path)

    @property
    def cfg(self):
        return self.module.cfg

    @property
    def output_size(self) -> int:
        return self.module.output_size

    @property
    def layer_ind_for_grads(self):
        return self.module.layer_ind_for_grads

    @layer_ind_for_grads.setter
    def layer_ind_for_grads(self, value):
        self.module.layer_ind_for_grads = value

    @property
    def permuted_positional_embeddings(self):
        return self.module.permuted_positional_embeddings

    @permuted_positional_embeddings.setter
    def permuted_positional_embeddings(self, value):
        self.module.permuted_positional_embeddings = value

    @property
    def random_batch_idx(self):
        return self.module.random_batch_idx

    @random_batch_idx.setter
    def random_batch_idx(self, value):
        self.module.random_batch_idx = value

    def calc_weights_norms(self, original_weights: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.module.calc_weights_norms(original_weights)
