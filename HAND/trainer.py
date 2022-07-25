from typing import Tuple, List
import os

import numpy as np
import torch
from clearml import Logger
from torch import optim
from torch.utils.data import DataLoader

from HAND.eval_func import EvalFunction
from HAND.logger import log_scalar_dict
from HAND.models.model import OriginalModel, ReconstructedModel
from HAND.predictors.predictor import HANDPredictorBase
from HAND.logger import create_experiment_dir, log_scalar_list, compute_grad_norms
from HAND.loss.attention_loss import AttentionLossBase
from HAND.loss.reconstruction_loss import ReconstructionLossBase
from HAND.loss.distillation_loss import DistillationLossBase
from HAND.loss.task_loss import TaskLossBase
from HAND.optimization.optimizer import OptimizerFactory
from HAND.optimization.scheduler import GenericScheduler, LRSchedulerFactory
from HAND.options import TrainConfig


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

        learnable_weights_shapes = self.reconstructed_model.get_learnable_weights_shapes()
        indices, positional_embeddings = self.reconstructed_model.get_indices_and_positional_embeddings()
        positional_embeddings = [torch.stack(layer_emb).to(self.device) for layer_emb in positional_embeddings]

        training_step = 0
        layer_ind_for_grads = 0
        for epoch in range(self.config.epochs):
            for batch_ind, (batch, ground_truth) in enumerate(self.train_dataloader):
                batch, ground_truth = batch.to(self.device), ground_truth.to(self.device)
                optimizer.zero_grad()

                reconstructed_weights = []
                layer_ind_for_grads = np.random.randint(0, len(positional_embeddings))
                # layer_ind_for_grads = (layer_ind_for_grads + 1) % len(positional_embeddings)
                for layer_ind, (layer_positional_embeddings, layer_shape) in enumerate(zip(positional_embeddings,
                                                                                         learnable_weights_shapes)):
                    if layer_ind == layer_ind_for_grads:
                        layer_reconstructed_weights = self._predict_layer_weights(layer_positional_embeddings,
                                                                                  layer_shape)
                        layer_reconstructed_weights.retain_grad()
                    else:
                        with torch.no_grad():
                            layer_reconstructed_weights = self._predict_layer_weights(layer_positional_embeddings,
                                                                                      layer_shape)
                    reconstructed_weights.append(layer_reconstructed_weights)

                self.reconstructed_model.update_weights(reconstructed_weights)

                original_outputs, original_feature_maps = self.original_model.get_feature_maps(batch)
                reconstructed_outputs, reconstructed_feature_maps = self.reconstructed_model.get_feature_maps(batch)

                # Compute reconstruction loss
                original_weights = self.original_model.get_learnable_weights()
                reconstruction_term = self.config.hand.reconstruction_loss_weight * self.reconstruction_loss(
                    reconstructed_weights, original_weights)

                if self._loss_warmup(epoch):
                    task_term, attention_term, distillation_term = 0, 0, 0
                else:
                    # Compute task loss
                    task_term = self.config.hand.task_loss_weight * self.task_loss(reconstructed_outputs, ground_truth)

                    # Compute attention loss
                    attention_term = self.config.hand.attention_loss_weight * self.attention_loss(
                        reconstructed_feature_maps, original_feature_maps)

                    # Compute distillation loss
                    distillation_term = self.config.hand.distillation_loss_weight * self.distillation_loss(
                        reconstructed_outputs, original_outputs)

                loss = reconstruction_term + task_term + attention_term + distillation_term
                loss.backward()

                if batch_ind % self.config.logging.log_interval == 0 and not self.config.logging.disable_logging:
                    loss_dict = dict(loss=loss,
                                     original_task_loss=task_term,
                                     reconstruction_loss=reconstruction_term,
                                     attention_loss=attention_term,
                                     distillation_loss=distillation_term)
                    self._log_training(training_step, reconstructed_weights, loss_dict, self.config.logging.verbose)
                    log_scalar_dict(dict(learning_rate=scheduler.get_last_lr()[-1]),
                                    title="learning_rate",
                                    iteration=epoch,
                                    logger=self.logger)

                self._clip_grad_norm()
                optimizer.step()
                scheduler.step_batch()
                training_step += 1

            scheduler.step_epoch()

            if epoch % self.config.eval_epochs_interval == 0:
                self.original_task_eval_fn.eval(self.reconstructed_model, self.test_dataloader, epoch, self.logger)

            if epoch % self.config.save_epoch_interval == 0:
                exp_dir = create_experiment_dir(self.config.logging.log_dir, self.config.exp_name)
                torch.save(self.predictor, os.path.join(exp_dir, f'hand_{epoch}.pth'))

    def _predict_layer_weights(self, layer_positional_embeddings, layer_shape):
        layer_reconstructed_weights = []
        for weights_batch_idx in range(0, layer_positional_embeddings.shape[0],
                                       self.config.hand.weights_batch_size):
            weights_batch = layer_positional_embeddings[
                            weights_batch_idx: weights_batch_idx + self.config.hand.weights_batch_size]
            layer_reconstructed_weights.append(self.predictor(weights_batch))
        layer_reconstructed_weights = torch.vstack(layer_reconstructed_weights).reshape(layer_shape)
        return layer_reconstructed_weights

    def _clip_grad_norm(self):
        if self.config.optim.max_gradient_norm is not None:
            for predictor_param in self.predictor.parameters():
                torch.nn.utils.clip_grad_norm_(predictor_param, max_norm=self.config.optim.max_gradient_norm)

    def _loss_warmup(self, epoch: int):
        return epoch < self.config.loss_warmup_epochs

    def _log_training(self, epoch: int, reconstructed_weights: List[torch.Tensor], loss_dict: dict, verbose: bool):
        log_scalar_dict(loss_dict,
                        title='training_loss',
                        iteration=epoch,
                        logger=self.logger)
        print(f'\nTraining loss is: {loss_dict["loss"]}')

        # logging norms
        if verbose is True:
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

    def _initialize_optimizer_and_scheduler(self) -> Tuple[torch.optim.Optimizer, GenericScheduler]:
        parameters_for_optimizer = list(self.predictor.parameters())
        if self.config.learn_fc_layer is True:
            parameters_for_optimizer.append(self.reconstructed_model.get_fully_connected_weights())

        optimizer = OptimizerFactory.get(parameters_for_optimizer, self.config)
        scheduler = LRSchedulerFactory.get(optimizer, len(self.train_dataloader), self.config)

        return optimizer, scheduler
