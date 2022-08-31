import numpy as np
import pyrallis
import torch
import matplotlib.pyplot as plt

from HAND.HAND_train import load_original_model
from HAND.eval_func import EvalFunction
from HAND.predictors.factory import HANDPredictorFactory, PredictorDataParallel
from HAND.tasks.dataloader_factory import DataloaderFactory
from HAND.tasks.pruning.prune_options import PruneConfig
from HAND.tasks.pruning.pruner import Pruner

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

    pruner = Pruner(cfg, predictor, original_model, reconstructed_model, device)
    reconstruction_pruned_model = pruner.prune(cfg.pruning_factor)
    magnitude_pruned_model = pruner.magnitude_prune(0.5)
    print(f'total weight number: {get_num_weights(original_model.get_learnable_weights())}\n')
    print(f'tzero weights: {get_num_zero_weights(magnitude_pruned_model.get_learnable_weights())}\n')


    # pruning_factors = np.linspace(0, 0.2, 20)
    # pruned_models_accuracies = list()
    # for pruning_factor in pruning_factors:
    #     pruned_model = pruner.prune(pruning_factor)
    #     pruned_model_accuracy = eval_fn.eval(pruned_model, test_dataloader, 0, None, '')
    #     pruned_models_accuracies.append(pruned_model_accuracy)
    #
    # plt.plot(pruning_factors, pruned_models_accuracies)
    # plt.xlabel('Pruning Factors')
    # plt.ylabel('Pruned Models Accuracies')
    # plt.show()

    # evaluate pruned model without fine-tuning
    print('evaluating reconstruction pruned model')
    pruned_model_accuracy = eval_fn.eval(reconstruction_pruned_model, test_dataloader, 0, None, '')
    print('evaluating magnitude pruned model')
    pruned_model_accuracy = eval_fn.eval(magnitude_pruned_model, test_dataloader, 0, None, '')
    print('evaluating original model')
    original_model_accuracy = eval_fn.eval(original_model, test_dataloader, 0, None, '')


if __name__ == '__main__':
    main()