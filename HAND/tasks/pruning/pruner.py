import json
import os

import pyrallis
import torch
from clearml import Task

from HAND.options import TrainConfig
from HAND.predictors.predictor import HANDPredictorFactory
from HAND.tasks.model_factory import ModelFactory
from HAND.tasks.pruning.prune_options import PruneConfig


def load_original_model(cfg: TrainConfig, device: torch.device):
    model_kwargs_path = cfg.original_model_path.replace('pt', 'json')
    if os.path.exists(model_kwargs_path):
        with open(model_kwargs_path) as f:
            model_kwargs = json.load(f)
    else:
        model_kwargs = dict()
    original_model, reconstructed_model = ModelFactory.get(cfg, device, **model_kwargs)
    return original_model, reconstructed_model

# def get_reconstruction_errors(reconstructed_weights, original_weights):
#
#     return errors
@pyrallis.wrap()
def main(cfg: PruneConfig):
    use_cuda = not cfg.train_cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    original_model, reconstructed_model = load_original_model(cfg.train_cfg, device)

    pos_embedding = reconstructed_model.positional_encoder.output_size
    predictor = HANDPredictorFactory(cfg.train_cfg.hand, input_size=pos_embedding).get_predictor().to(device)
    predictor.load_state_dict(torch.load(cfg.predictor_path).state_dict())

    learnable_weights_shapes = reconstructed_model.get_learnable_weights_shapes()
    indices, positional_embeddings = reconstructed_model.get_indices_and_positional_embeddings()
    positional_embeddings = [torch.stack(layer_emb).to(device) for layer_emb in positional_embeddings]

    original_weights = original_model.get_learnable_weights()

    reconstructed_weights = predictor.predict_all(positional_embeddings,
                                                  original_weights,
                                                  learnable_weights_shapes)
    reconstructed_model.update_weights(reconstructed_weights)



    # import original model
    # impoet predicotr and predict weights
    # calculate reconstruction error
    # flat all errors of the network and sort (or get indexes of top precentile numpy)
    # zero weights
    # evaluate pruned model without fine tuning


if __name__ == '__main__':
    main()
