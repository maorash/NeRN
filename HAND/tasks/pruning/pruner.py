from typing import List

import copy
import pyrallis
import torch

from HAND.HAND_train import load_original_model
from HAND.eval_func import EvalFunction
from HAND.predictors.factory import HANDPredictorFactory, PredictorDataParallel
from HAND.predictors.predictor import HANDPredictorBase
from HAND.tasks.dataloader_factory import DataloaderFactory
from HAND.tasks.model_factory import ModelFactory
from HAND.tasks.pruning.prune_options import PruneConfig


def get_reconstruction_errors(reconstructed_weights, original_weights):
    reconstruction_errors = list()
    for original_weight, reconstructed_weight in zip(original_weights, reconstructed_weights):
        error = (original_weight - reconstructed_weight) ** 2
        reconstruction_errors.append(error)
    return reconstruction_errors


def get_largest_error_indices(reconstruction_errors: List[torch.Tensor], pruning_factor: float):
    all_sorted, all_sorted_idx = torch.sort(torch.cat([-1 * t.view(-1) for t in reconstruction_errors]))
    cum_num_elements = torch.cumsum(torch.tensor([t.numel() for t in reconstruction_errors]), dim=0)
    cum_num_elements = torch.cat([torch.tensor([0]), cum_num_elements])

    n = int(cum_num_elements[-1].item() * pruning_factor)
    split_indices_lt = [all_sorted_idx[:n] < cum_num_elements[i + 1] for i, _ in enumerate(cum_num_elements[1:])]
    split_indices_ge = [all_sorted_idx[:n] >= cum_num_elements[i] for i, _ in enumerate(cum_num_elements[:-1])]
    largest_error_indices = [all_sorted_idx[:n][torch.logical_and(lt, ge)] - c for lt, ge, c in
                             zip(split_indices_lt, split_indices_ge, cum_num_elements[:-1])]

    # n_largest_errors = [t.view(-1)[idx] for t, idx in zip(reconstruction_errors, smallest_error_indices)]

    # returns list of tensors with linear indices in each tensor
    return largest_error_indices


def prune_weights(original_weights: List[torch.Tensor], indices_to_prune: List[torch.Tensor]):
    with torch.no_grad():
        pruned_weights = copy.deepcopy(original_weights)
        for original_layer, layer_indices_to_prune in zip(pruned_weights, indices_to_prune):
            original_layer.view(-1)[layer_indices_to_prune] = 0
    return pruned_weights


@pyrallis.wrap()
def main(cfg: PruneConfig):
    use_cuda = not cfg.train_cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # import original model
    original_model, reconstructed_model = load_original_model(cfg.train_cfg, device)

    # import predictor
    pos_embedding = reconstructed_model.positional_encoder.output_size
    predictor = HANDPredictorFactory(cfg.train_cfg, input_size=pos_embedding).get_predictor().to(device)
    if cfg.is_data_parallel:
        predictor = PredictorDataParallel(predictor)
        predictor.load_state_dict(torch.load(cfg.predictor_path).state_dict())
        predictor = predictor.module
    else:
        predictor.load_state_dict(torch.load(cfg.predictor_path).state_dict())

    learnable_weights_shapes = reconstructed_model.get_learnable_weights_shapes()
    indices, positional_embeddings = reconstructed_model.get_indices_and_positional_embeddings()
    positional_embeddings = [torch.stack(layer_emb).to(device) for layer_emb in positional_embeddings]

    # get original weights and predict reconstructed weights
    original_weights = original_model.get_learnable_weights()
    reconstructed_weights = HANDPredictorBase.predict_all(predictor, positional_embeddings,
                                                          original_weights,
                                                          learnable_weights_shapes)
    reconstructed_model.update_weights(reconstructed_weights)
    updated_reconstructed_weights = reconstructed_model.get_learnable_weights()

    # calculate reconstruction error
    reconstruction_errors = get_reconstruction_errors(updated_reconstructed_weights, original_weights)

    # get indices of weights to prune - those with the largest reconstruction  errors
    largest_error_indices = get_largest_error_indices(reconstruction_errors, cfg.pruning_factor)

    # prune original weights
    pruned_original_weights = prune_weights(original_weights, largest_error_indices)

    # create new model with pruned weights
    pruned_model = ModelFactory.models[cfg.train_cfg.task.original_model_name][1](original_model,
                                                                                  cfg.train_cfg.hand.embeddings,
                                                                                  sampling_mode=cfg.train_cfg.hand.sampling_mode).to(device)
    pruned_model.update_weights(pruned_original_weights)

    # evaluate pruned model without fine-tuning
    _, test_dataloader = DataloaderFactory.get(cfg.train_cfg.task, **{'batch_size': cfg.train_cfg.batch_size,
                                                                      'num_workers': cfg.train_cfg.num_workers})
    eval_fn = EvalFunction(cfg.train_cfg)
    pruned_model_accuracy = eval_fn.eval(pruned_model, test_dataloader, 0, None, '')
    original_model_accuracy = eval_fn.eval(original_model, test_dataloader, 0, None, '')
    # the accuracys both equal to 69%. why is the original model changing ??? it was on 93%. happens in line 85 somehow
if __name__ == '__main__':
    main()
