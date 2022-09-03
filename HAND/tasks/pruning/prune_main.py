import numpy as np
import pyrallis
import torch
import matplotlib.pyplot as plt


from HAND.HAND_train import load_original_model
from HAND.eval_func import EvalFunction
from HAND.predictors.factory import HANDPredictorFactory, PredictorDataParallel
from HAND.predictors.predictor import HANDPredictorBase
from HAND.tasks.dataloader_factory import DataloaderFactory
from HAND.tasks.model_factory import ModelFactory
from HAND.tasks.pruning.prune_options import PruneConfig
from HAND.tasks.pruning.pruner import Pruner
from HAND.permutations import utils as permutations_utils


def get_num_zero_weights(weight_list):
    num_zeros = 0
    for layer in weight_list:
        num_zeros += torch.sum(layer == 0)
    return num_zeros


def get_num_weights(weight_list):
    num_weights = 0
    for layer in weight_list:
        num_weights += torch.numel(layer)
    return num_weights


@pyrallis.wrap()
def main(cfg: PruneConfig):
    use_cuda = not cfg.train_cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # import original model
    original_model, reconstructed_model = load_original_model(cfg.train_cfg, device)

    _, test_dataloader = DataloaderFactory.get(cfg.train_cfg.task,
                                               **{'batch_size': cfg.train_cfg.batch_size,
                                                  'num_workers': cfg.train_cfg.num_workers})
    eval_fn = EvalFunction(cfg.train_cfg)

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
    positional_embeddings = permutations_utils.permute(positional_embeddings,
                                                       original_weights,
                                                       cfg.train_cfg.hand.permute_mode)
    reconstructed_weights = HANDPredictorBase.predict_all(predictor, positional_embeddings,
                                                          original_weights,
                                                          learnable_weights_shapes)
    reconstructed_model.update_weights(reconstructed_weights)

    pruned_model = ModelFactory.models[cfg.train_cfg.task.original_model_name][1](original_model,
                                                                                           cfg.train_cfg.hand.embeddings,
                                                                                           sampling_mode=cfg.train_cfg.hand.sampling_mode).to(
            device)

    pruner = Pruner(cfg, predictor, original_model, reconstructed_model, pruned_model, device)

    # absolute_nern_pruned_model = pruner.prune(0.6, absolute=True)
    # relative_nern_pruned_model
    pruner.prune(0.5, errer_metric='relative')
    eval_fn.eval(pruned_model, test_dataloader, 0, None, '')
    # magnitude_pruned_model
    pruner.magnitude_prune(0.5)
    eval_fn.eval(pruned_model, test_dataloader, 0, None, '')
    # random_pruned_model
    pruner.random_prune(0.5)
    eval_fn.eval(pruned_model, test_dataloader, 0, None, '')

    # print(f'total weight number: {get_num_weights(original_model.get_learnable_weights())}\n')
    # print(f'tzero weights: {get_num_zero_weights(magnitude_pruned_model.get_learnable_weights())}\n')

    pruning_factors = np.linspace(0, 1, 10)
    relative_reconstruction_pruned_accuracies = list()
    # relative_squared_reconstruction_pruned_accuracies = list()
    magnitude_pruned_accuracies = list()
    random_pruned_accuracies = list()

    # for pruning_factor in pruning_factors:
    #     pruner.prune(pruning_factor, error_metric='relative')
    #     relative_reconstruction_accuracy = eval_fn.eval(pruned_model, test_dataloader, 0, None, '')
    #
    #     # pruner.prune(pruning_factor, error_metric='relative_squared')
    #     # relative_squared_reconstruction_accuracy = eval_fn.eval(pruned_model, test_dataloader, 0, None, '')
    #
    #     pruner.magnitude_prune(pruning_factor)
    #     magnitude_accuracy = eval_fn.eval(pruned_model, test_dataloader, 0, None, '')
    #
    #     pruner.random_prune(pruning_factor)
    #     random_accuracy = eval_fn.eval(pruned_model, test_dataloader, 0, None, '')
    #
    #     relative_reconstruction_pruned_accuracies.append(relative_reconstruction_accuracy)
    #     # relative_squared_reconstruction_pruned_accuracies.append(relative_squared_reconstruction_accuracy)
    #     magnitude_pruned_accuracies.append(magnitude_accuracy)
    #     random_pruned_accuracies.append(random_accuracy)
    #
    # plt.plot(pruning_factors, relative_reconstruction_pruned_accuracies, label='relative_recon_error')
    # plt.plot(pruning_factors, relative_squared_reconstruction_pruned_accuracies, label='relative_squared_recon_error')
    # plt.plot(pruning_factors, magnitude_pruned_accuracies, label='magnitude')
    # # plt.plot(pruning_factors, random_pruned_accuracies, label='random')
    # plt.legend()
    # plt.xlabel('Pruning Factors')
    # plt.ylabel('Pruned Models Accuracies')
    # plt.title('93% Reconstruction')
    # plt.show()

    #evaluate pruned model without fine-tuning
    # print('evaluating absolute nern pruned model')
    # eval_fn.eval(absolute_nern_pruned_model, test_dataloader, 0, None, '')
    # print(f'evaluating relative nern pruned model: {get_num_zero_weights(relative_nern_pruned_model.get_learnable_weights())} zeros')
    # eval_fn.eval(relative_nern_pruned_model, test_dataloader, 0, None, '')
    # print(f'evaluating magnitude pruned model: {get_num_zero_weights(magnitude_pruned_model.get_learnable_weights())} zeros')
    # eval_fn.eval(magnitude_pruned_model, test_dataloader, 0, None, '')
    # print('evaluating random pruned model')
    # eval_fn.eval(random_pruned_model, test_dataloader, 0, None, '')
    # print('evaluating original model')
    # eval_fn.eval(original_model, test_dataloader, 0, None, '')
    # print('evaluating reconstructed model')
    # eval_fn.eval(reconstructed_model, test_dataloader, 0, None, '')
    #

if __name__ == '__main__':
    main()
