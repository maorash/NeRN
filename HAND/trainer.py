import os

import torch
from clearml import Logger
from torch import optim

from HAND.eval_func import EvalFunction
from HAND.models.model import OriginalModel, ReconstructedModel
from HAND.predictors.predictor import HANDPredictorBase
from HAND.tasks.mnist.mnist_train_main import get_dataloaders
from logger import create_experiment_dir
from loss import ReconstructionLoss, FeatureMapsDistillationLoss, OutputDistillationLoss
from options import TrainConfig


class Trainer:
    def __init__(self, config: TrainConfig,
                 predictor: HANDPredictorBase,
                 reconstruction_loss: ReconstructionLoss,
                 feature_maps_distillation_loss: FeatureMapsDistillationLoss,
                 output_distillation_loss: OutputDistillationLoss,
                 original_model: OriginalModel,
                 reconstructed_model: ReconstructedModel,
                 original_task_eval_fn: EvalFunction,
                 logger: Logger,
                 device):
        self.config = config
        self.predictor = predictor
        self.reconstruction_loss = reconstruction_loss
        self.feature_maps_distillation_loss = feature_maps_distillation_loss
        self.output_distillation_loss = output_distillation_loss
        self.original_model = original_model
        self.reconstructed_model = reconstructed_model
        self.original_task_eval_fn = original_task_eval_fn
        self.device = device
        self.logger = logger

    def train(self):
        self._set_grads_for_training()

        exp_dir = create_experiment_dir(self.config.logging.log_dir, self.config.exp_name)
        optimizer = self._initialize_optimizer()

        data_kwargs = {'batch_size': self.config.batch_size}
        test_dataloader, train_dataloader = get_dataloaders(test_kwargs=data_kwargs,
                                                            train_kwargs=data_kwargs)

        # For a number of epochs
        for epoch in range(self.config.epochs):
            indices, positional_embeddings = self.reconstructed_model.get_indices_and_positional_embeddings()
            predicted_weights = []
            for index, positional_embedding in zip(indices, positional_embeddings):
                # Reconstruct all of the original model's weights using the predictors model
                reconstructed_weights = self.predictor(positional_embedding.to(self.device))  # TODO: clean this up
                predicted_weights.append(reconstructed_weights)

            new_weights = self.reconstructed_model.aggregate_predicted_weights(predicted_weights)
            self.reconstructed_model.update_whole_weights(new_weights)

            # Now we can see how good our reconstructed model is
            original_weights = self.original_model.get_learnable_weights()
            reconstruction_term = self.config.hand.reconstruction_loss_weight * self.reconstruction_loss(
                new_weights, original_weights)
            feature_maps_term = 0.
            # feature_maps_term = self.config.hand.feature_maps_distillation_loss_weight * self.feature_maps_distillation_loss(
            #     batch, self.reconstructed_model, self.original_model)  # TODO: where does the batch come from? which loop

            outputs_term = 0.
            # outputs_term = self.config.hand.output_distillation_loss_weight * self.output_distillation_loss(
            #     self.reconstructed_model, self.original_model)# TODO: where does the batch come from? which loop

            loss = reconstruction_term + feature_maps_term + outputs_term
            if epoch % self.config.logging.log_interval == 0:
                self.logger.report_scalar('training_loss', 'training_loss', loss, epoch)
                print(f'\nTraining loss is: {loss}')
            loss.backward()
            optimizer.step()

            if epoch % self.config.eval_epochs_interval == 0:
                self.original_task_eval_fn.eval(self.reconstructed_model, test_dataloader, epoch, self.logger)

            if epoch % self.config.save_epoch_interval == 0:
                torch.save(self.predictor, os.path.join(exp_dir, f'hand_{epoch}.pth'))

    def _set_grads_for_training(self):
        self.original_model.eval()
        self.reconstructed_model.eval()
        for param in self.original_model.parameters():
            param.requires_grad = False
        for param in self.reconstructed_model.parameters():
            param.requires_grad = False

        self.predictor.train()
        for param in self.predictor.parameters():
            param.requires_grad = True

    def _initialize_optimizer(self):
        optimizer_type = getattr(optim, self.config.optimizer)
        if self.config.optimizer != "SGD":
            optimizer = optimizer_type(self.predictor.parameters(),
                                       betas=self.config.betas, lr=self.config.lr)
        else:
            optimizer = optimizer_type(self.predictor.parameters(), lr=self.config.lr)
        return optimizer
