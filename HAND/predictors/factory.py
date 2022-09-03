from abc import abstractmethod
from typing import List

import torch
from torch.nn import DataParallel

from HAND.options import TrainConfig
from HAND.predictors.predictor import HANDPredictorBase
from HAND.predictors.predictor import HANDBasicPredictor, HANDKxKPredictor, HANDKxKNerFPredictor, \
    HANDKxKResidualPredictor


class HANDPredictorFactory:
    def __init__(self, cfg: TrainConfig, input_size: int):
        self.cfg = cfg
        self.input_size = input_size

    def get_predictor(self):
        if self.cfg.hand.method == 'basic':
            predictor = HANDBasicPredictor(self.cfg.hand, self.input_size)
        elif self.cfg.hand.method == 'kxk':
            predictor = HANDKxKPredictor(self.cfg.hand, self.input_size)
        elif self.cfg.hand.method == 'kxk_nerf':
            predictor = HANDKxKNerFPredictor(self.cfg.hand, self.input_size)
        elif self.cfg.hand.method == 'kxk_residual':
            predictor = HANDKxKResidualPredictor(self.cfg.hand, self.input_size)
        else:
            raise ValueError(f'Not recognized predictor type {self.cfg.hand.method}')

        if self.cfg.num_gpus > 1:
            predictor = PredictorDataParallel(predictor, device_ids=list(range(self.cfg.num_gpus)))

        return predictor


class PredictorDataParallel(DataParallel):
    def __init__(self, module: HANDPredictorBase, *args, **kwargs):
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
