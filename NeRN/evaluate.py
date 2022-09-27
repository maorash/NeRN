import json
import os

import pyrallis
import torch

from NeRN.options import Config
from NeRN.predictors.factory import NeRNPredictorFactory
from NeRN.tasks.model_factory import ModelFactory
from NeRN.predictors.predictor import NeRNPredictorBase
from NeRN.eval_func import EvalFunction
from NeRN.tasks.dataloader_factory import DataloaderFactory


def load_original_model(cfg: Config, device: torch.device):
    model_kwargs_path = cfg.original_model_path.replace('pt', 'json')
    if os.path.exists(model_kwargs_path):
        with open(model_kwargs_path) as f:
            model_kwargs = json.load(f)
    else:
        model_kwargs = dict()
    original_model, reconstructed_model = ModelFactory.get(cfg, device, **model_kwargs)
    return original_model, reconstructed_model


@pyrallis.wrap()
def main(cfg: Config):
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    original_model, reconstructed_model = load_original_model(cfg, device)

    pos_embedding = reconstructed_model.output_size
    predictor = NeRNPredictorFactory(cfg, input_size=pos_embedding).get_predictor().to(device)

    init_predictor(cfg, predictor)

    _, test_dataloader = DataloaderFactory.get(cfg.task, **{'batch_size': cfg.batch_size, 'num_workers': cfg.num_workers})

    learnable_weights_shapes = reconstructed_model.get_learnable_weights_shapes()
    indices, positional_embeddings = reconstructed_model.get_indices_and_positional_embeddings()
    reconstructed_weights = NeRNPredictorBase.predict_all(predictor,
                                                          positional_embeddings,
                                                          original_model.get_learnable_weights(),
                                                          learnable_weights_shapes)

    reconstructed_model.update_weights(reconstructed_weights)

    eval_func = EvalFunction(cfg)
    eval_func.eval(reconstructed_model, test_dataloader, iteration=0, logger=None)


def init_predictor(cfg, predictor):
    if cfg.nern.init != "checkpoint":
        raise ValueError("Please pass a checkpoint to init the predictor")
    print(f"Loading pretrained weights from: {cfg.nern.checkpoint_path}")
    predictor.load(cfg.nern.checkpoint_path)


if __name__ == '__main__':
    main()
