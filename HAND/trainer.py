from typing import Tuple
import os

import torch
from clearml import Logger
from torch import optim
from torch.utils.data import DataLoader

from HAND.eval_func import EvalFunction
from HAND.logger import log_scalar_dict
from HAND.models.model import OriginalModel, ReconstructedModel
from HAND.predictors.predictor import HANDPredictorBase
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
                 task_dataloaders: Tuple[DataLoader, DataLoader],
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
        self.test_dataloader, self.train_dataloader = task_dataloaders
        self.logger = logger

    def train(self):
        self._set_grads_for_training()

        optimizer, scheduler = self._initialize_optimizer_and_scheduler()

        learnable_weights_shapes = [weights.shape for weights in self.reconstructed_model.get_learnable_weights()]
        indices, positional_embeddings = self.reconstructed_model.get_indices_and_positional_embeddings()
        positional_embeddings = [torch.stack(layer_emb).to(self.device) for layer_emb in positional_embeddings]

        training_step = 0
        for epoch in range(self.config.epochs):
            for batch_idx, (batch, ground_truth) in enumerate(self.train_dataloader):
                batch, ground_truth = batch.to(self.device), ground_truth.to(self.device)
                optimizer.zero_grad()

                reconstructed_weights = []
                # Each forward pass of the prediction model predicts an entire layer's weights
                for layer_positional_embeddings, layer_shape in zip(positional_embeddings, learnable_weights_shapes):
                    layer_reconstructed_weights = self.predictor(layer_positional_embeddings).reshape(layer_shape)
                    layer_reconstructed_weights.retain_grad()
                    reconstructed_weights.append(layer_reconstructed_weights)

                self.reconstructed_model.update_weights(reconstructed_weights)

                original_outputs, original_feature_maps = self.original_model.get_feature_maps(batch)
                reconstructed_outputs, reconstructed_feature_maps = self.reconstructed_model.get_feature_maps(batch)

                # Compute reconstruction loss
                original_weights = self.original_model.get_learnable_weights()
                reconstruction_term = self.config.hand.reconstruction_loss_weight * self.reconstruction_loss(
                    reconstructed_weights, original_weights)

                # Compute task loss
                task_term = self.config.hand.task_loss_weight * self.task_loss(reconstructed_outputs, ground_truth)

                # Compute attention loss
                attention_term = self.config.hand.attention_loss_weight * self.attention_loss(
                    reconstructed_feature_maps, original_feature_maps)

                # Compute distillation loss
                distillation_term = self.config.hand.distillation_loss_weight * self.distillation_loss(
                    reconstructed_outputs, original_outputs)

                loss = task_term + reconstruction_term + attention_term + distillation_term
                loss.backward()

                if batch_idx % self.config.logging.log_interval == 0 and not self.config.logging.disable_logging and epoch > 0:
                    loss_dict = dict(loss=loss,
                                     original_task_loss=task_term,
                                     reconstruction_loss=reconstruction_term,
                                     attention_loss=attention_term,
                                     distillation_loss=distillation_term)
                    self._log_training(training_step, reconstructed_weights, loss_dict)
                    log_scalar_dict(dict(learning_rate=scheduler.get_last_lr()[-1]),
                                    title="learning_rate",
                                    iteration=epoch,
                                    logger=self.logger)

                optimizer.step()
                scheduler.step()

                training_step += 1

            if epoch % self.config.eval_epochs_interval == 0:
                self.original_task_eval_fn.eval(self.reconstructed_model, self.test_dataloader, epoch, self.logger)

            if epoch % self.config.save_epoch_interval == 0:
                exp_dir = create_experiment_dir(self.config.logging.log_dir, self.config.exp_name)
                torch.save(self.predictor, os.path.join(exp_dir, f'hand_{epoch}.pth'))

    def _log_training(self, epoch, reconstructed_weights, loss_dict: dict):
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

        reconstructed_weights_grad_norms = compute_grad_norms(reconstructed_weights)
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

    def _initialize_optimizer_and_scheduler(self):
        optimizer_type = getattr(optim, self.config.optimizer)
        parameters_for_optimizer = list(self.predictor.parameters())
        if self.config.learn_fc_layer is True:
            parameters_for_optimizer.append(self.reconstructed_model.get_fully_connected_weights())
        if self.config.optimizer != "SGD":
            optimizer = optimizer_type(parameters_for_optimizer,
                                       betas=self.config.betas, lr=self.config.lr)
        else:
            optimizer = optimizer_type(self.predictor.parameters(), lr=self.config.lr)

        if self.config.lr_decay_type is not None:
            scheduler_type = getattr(optim.lr_scheduler, self.config.lr_decay_type)
            scheduler = scheduler_type(optimizer, T_max=(self.config.epochs * len(self.train_dataloader)),
                                       eta_min=1e-6)
        else:
            scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=0, last_epoch=0)

        return optimizer, scheduler
