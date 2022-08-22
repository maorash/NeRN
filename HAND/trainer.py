from typing import Tuple, List
import os

import torch
from clearml import Logger
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
        self.train_dataloader, self.test_dataloader = task_dataloaders
        self.logger = logger
        self.exp_dir_path = None
        self.max_eval_accuracy = 0
        self.loss_window = []

    def train(self):
        self._set_grads_for_training()
        self._set_training_steps()

        optimizer, scheduler = self._initialize_optimizer_and_scheduler()

        learnable_weights_shapes = self.reconstructed_model.get_learnable_weights_shapes()
        indices, positional_embeddings = self.reconstructed_model.get_indices_and_positional_embeddings()
        positional_embeddings = [torch.stack(layer_emb).to(self.device) for layer_emb in positional_embeddings]

        self.exp_dir_path = create_experiment_dir(self.config.logging.log_dir, self.config.logging.exp_name)

        original_weights = self.original_model.get_learnable_weights()

        training_step = 0
        epoch = 0

        while True:
            for batch, ground_truth in self.train_dataloader:
                batch, ground_truth = batch.to(self.device), ground_truth.to(self.device)
                optimizer.zero_grad()

                reconstructed_weights = self.predictor.predict_all(positional_embeddings,
                                                                   original_weights,
                                                                   learnable_weights_shapes)
                self.reconstructed_model.update_weights(reconstructed_weights)

                original_outputs, original_feature_maps = self.original_model.get_feature_maps(batch)
                reconstructed_outputs, reconstructed_feature_maps = self.reconstructed_model.get_feature_maps(batch)

                # Compute reconstruction loss
                reconstruction_term = self.config.hand.reconstruction_loss_weight * self.reconstruction_loss(
                    reconstructed_weights, original_weights)

                if self._loss_warmup(training_step):
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

                if training_step % self.config.logging.log_interval == 0 and not self.config.logging.disable_logging:
                    loss_dict = dict(loss=loss,
                                     original_task_loss=task_term,
                                     reconstruction_loss=reconstruction_term,
                                     attention_loss=attention_term,
                                     distillation_loss=distillation_term)
                    self._log_training(training_step, reconstructed_weights, loss_dict, scheduler.get_last_lr(),
                                       self.config.logging.verbose)

                self._clip_grad_norm()
                optimizer.step()
                scheduler.check_and_step(training_step)
                training_step += 1

                if training_step % self.config.eval_loss_window_interval == 0:
                    self._add_to_loss_window(loss.item())
                    if len(self.loss_window) == self.config.eval_loss_window_size and loss <= min(self.loss_window):
                        self._eval(training_step, "greedy")
                if self.config.eval_iterations_interval is not None \
                        and training_step % self.config.eval_iterations_interval == 0:
                    self._eval(training_step)
                if self.config.save_iterations_interval is not None \
                        and training_step % self.config.save_iterations_interval == 0:
                    self._save_checkpoint(f"step_{training_step}")

                if training_step >= self.config.num_iterations:
                    return

            epoch += 1

            if self.config.eval_epochs_interval is not None and epoch % self.config.eval_epochs_interval == 0:
                self._eval(training_step)
            if self.config.save_epochs_interval is not None and epoch % self.config.save_epochs_interval == 0:
                self._save_checkpoint(f"epoch_{epoch}")

    def _eval(self, iteration, log_suffix=None):
        accuracy = self.original_task_eval_fn.eval(self.reconstructed_model, self.test_dataloader, iteration,
                                                   self.logger, log_suffix)
        if accuracy > self.max_eval_accuracy:
            self.max_eval_accuracy = accuracy
            self._save_checkpoint(f"best")

    def _add_to_loss_window(self, loss):
        if len(self.loss_window) == self.config.eval_loss_window_size:
            self.loss_window.pop(0)
        self.loss_window.append(loss)

    def _save_checkpoint(self, checkpoint_suffix: str):
        torch.save(self.predictor, os.path.join(self.exp_dir_path,
                                                f"hand_{self.config.logging.exp_name}_{checkpoint_suffix}.pth"))

    def _clip_grad_norm(self):
        if self.config.optim.max_gradient_norm is not None:
            for predictor_param in self.predictor.parameters():
                torch.nn.utils.clip_grad_norm_(predictor_param, max_norm=self.config.optim.max_gradient_norm)
        if self.config.optim.max_gradient is not None:
            for predictor_param in self.predictor.parameters():
                torch.nn.utils.clip_grad_value_(predictor_param, clip_value=self.config.optim.max_gradient)

    def _loss_warmup(self, training_step: int):
        return training_step < self.config.loss_warmup_iterations

    def _log_training(self, training_step: int,
                      reconstructed_weights: List[torch.Tensor],
                      loss_dict: dict,
                      lr: float,
                      verbose: bool):
        log_scalar_dict(loss_dict,
                        title='training_loss',
                        iteration=training_step,
                        logger=self.logger)
        log_scalar_dict(dict(learning_rate=lr),
                        title="learning_rate",
                        iteration=training_step,
                        logger=self.logger)
        print(f'\nTraining loss is: {loss_dict["loss"]}')

        # logging norms
        if verbose is True:
            self._log_training_verbose(reconstructed_weights, training_step)

    def _log_training_verbose(self, reconstructed_weights, training_step):
        original_weights_norms = self.original_model.get_learnable_weights_norms()
        log_scalar_list(original_weights_norms,
                        title='weight_norms',
                        series_name='original',
                        iteration=training_step,
                        logger=self.logger)
        reconstructed_weights_norms = self.reconstructed_model.get_learnable_weights_norms()
        log_scalar_list(reconstructed_weights_norms,
                        title='weight_norms',
                        series_name='reconstructed',
                        iteration=training_step,
                        logger=self.logger)
        reconstructed_weights_grad_norms = compute_grad_norms(reconstructed_weights)
        log_scalar_list(reconstructed_weights_grad_norms,
                        title='reconstructed_weights_grad_norms',
                        series_name='layer',
                        iteration=training_step,
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

    def _set_training_steps(self):
        if self.config.num_iterations is None:
            self.config.num_iterations = self.config.epochs * len(self.train_dataloader)
