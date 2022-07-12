import os

import torch
from clearml import Logger
from torch import optim

from HAND.eval_func import EvalFunction
from HAND.logger import set_grads_for_logging, log_scalar_dict
from HAND.models.model import OriginalModel, ReconstructedModel
from HAND.predictors.predictor import HANDPredictorBase
from HAND.tasks.mnist.mnist_train_main import get_dataloaders
from logger import create_experiment_dir, log_scalar_list, compute_grad_norms
from loss.attention_loss import AttentionLossBase
from loss.reconstruction_loss import ReconstructionLossBase
from loss.distillation_loss import DistillationLossBase
from loss.task_loss import TaskLossBase
from options import TrainConfig


class Trainer:
    def __init__(self, config: TrainConfig,
                 predictor: HANDPredictorBase,
                 task_loss: TaskLossBase,
                 reconstruction_loss: ReconstructionLossBase,
                 attention_loss: AttentionLossBase,
                 distillation_loss: DistillationLossBase,
                 original_model: OriginalModel,
                 reconstructed_model: ReconstructedModel,
                 original_task_eval_fn: EvalFunction,
                 logger: Logger,
                 device):
        self.config = config
        self.predictor = predictor
        self.task_loss = task_loss
        self.reconstruction_loss = reconstruction_loss
        self.attention_loss = attention_loss
        self.distillation_loss = distillation_loss
        self.original_model = original_model
        self.reconstructed_model = reconstructed_model
        self.original_task_eval_fn = original_task_eval_fn
        self.device = device
        self.logger = logger

    def train(self):
        self._set_grads_for_training()

        optimizer = self._initialize_optimizer()

        data_kwargs = {'batch_size': self.config.batch_size}
        test_dataloader, train_dataloader = get_dataloaders(test_kwargs=data_kwargs,
                                                            train_kwargs=data_kwargs)

        training_step = 0
        for epoch in range(self.config.epochs):
            for batch_idx, (batch, target) in enumerate(train_dataloader):
                batch, target = batch.to(self.device), target.to(self.device)
                optimizer.zero_grad()

                # Predict weights for the reconstructed model using HAND
                indices, positional_embeddings = self.reconstructed_model.get_indices_and_positional_embeddings()
                predicted_weights = []
                for index, positional_embedding in zip(indices, positional_embeddings):
                    # Reconstruct all of the original model's weights using the predictors model
                    reconstructed_weights = self.predictor(positional_embedding.to(self.device))  # TODO: clean this up
                    predicted_weights.append(reconstructed_weights)

                new_weights = self.reconstructed_model.aggregate_predicted_weights(predicted_weights)
                set_grads_for_logging(new_weights)

                self.reconstructed_model.update_whole_weights(new_weights)

                # Compute reconstruction loss
                original_weights = self.original_model.get_learnable_weights()
                reconstruction_term = self.config.hand.reconstruction_loss_weight * self.reconstruction_loss(
                    new_weights, original_weights)

                # Compute task loss
                reconstructed_model_predictions = self.reconstructed_model.reconstructed_model(batch)
                task_term = self.config.hand.task_loss_weight * self.task_loss(reconstructed_model_predictions, target)

                # Compute attention loss
                attention_term = self.config.hand.attention_loss_weight * \
                                 self.attention_loss(batch, self.reconstructed_model, self.original_model)

                # Compute distillation loss
                distillation_term = self.config.hand.distillation_loss_weight * \
                                    self.distillation_loss(batch, self.reconstructed_model, self.original_model)

                loss = task_term + reconstruction_term + attention_term + distillation_term
                loss.backward()

                if batch_idx % self.config.logging.log_interval == 0 and not self.config.logging.disable_logging:
                    loss_dict = dict(loss=loss,
                                     original_task_loss=task_term,
                                     reconstruction_loss=reconstruction_term,
                                     attention_loss=attention_term,
                                     distillation_loss=distillation_term)
                    self._log_training(training_step, new_weights, loss_dict)

                optimizer.step()
                training_step += 1

            if epoch % self.config.eval_epochs_interval == 0:
                self.original_task_eval_fn.eval(self.reconstructed_model, test_dataloader, epoch, self.logger)

            if epoch % self.config.save_epoch_interval == 0:
                exp_dir = create_experiment_dir(self.config.logging.log_dir, self.config.exp_name)
                torch.save(self.predictor, os.path.join(exp_dir, f'hand_{epoch}.pth'))

    def _log_training(self, epoch, new_weights, loss_dict: dict):
        log_scalar_dict(loss_dict,
                        title='training_loss',
                        iteration=epoch,
                        logger=self.logger)
        print(f'\nTraining loss is: {loss_dict["loss"]}')

        # logging norms
        original_weights_norms = self.original_model.get_learnable_weights_norms()
        log_scalar_list(original_weights_norms,
                        title='weight_norms',
                        series_name='original',
                        iteration=epoch,
                        logger=self.logger)

        reconstructed_weights_norms = self.reconstructed_model.get_learnable_weights_norms()
        log_scalar_list(reconstructed_weights_norms,
                        title='weight_norms',
                        series_name='reconstructed',
                        iteration=epoch,
                        logger=self.logger)

        reconstructed_weights_grad_norms = compute_grad_norms(new_weights)
        log_scalar_list(reconstructed_weights_grad_norms,
                        title='reconstructed_weights_grad_norms',
                        series_name='layer',
                        iteration=epoch,
                        logger=self.logger)

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
        parameters_for_optimizer = list(self.predictor.parameters())
        if self.config.learn_fc_layer is True:
            parameters_for_optimizer.append(self.reconstructed_model.get_fully_connected_weights())
        if self.config.optimizer != "SGD":
            optimizer = optimizer_type(parameters_for_optimizer,
                                       betas=self.config.betas, lr=self.config.lr)
        else:
            optimizer = optimizer_type(self.predictor.parameters(), lr=self.config.lr)

        return optimizer
