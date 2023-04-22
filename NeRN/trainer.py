from typing import Tuple, List, Union
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from clearml import Logger
from torch.utils.data import DataLoader

from NeRN.eval_func import EvalFunction
from NeRN.log_utils import log_scalar_dict
from NeRN.models.model import OriginalModel, OriginalDataParallel, ReconstructedModel, ReconstructedDataParallel
from NeRN.predictors.predictor import NeRNPredictorBase
from NeRN.predictors.factory import PredictorDataParallel
from NeRN.log_utils import create_experiment_dir, log_scalar_list, compute_grad_norms
from NeRN.loss.attention_loss import AttentionLossBase
from NeRN.loss.reconstruction_loss import ReconstructionLossBase
from NeRN.loss.distillation_loss import DistillationLossBase
from NeRN.optimization.optimizer import OptimizerFactory
from NeRN.optimization.scheduler import GenericScheduler, LRSchedulerFactory
from NeRN.options import Config


class Trainer:
    def __init__(self, config: Config,
                 predictor: Union[NeRNPredictorBase, PredictorDataParallel],
                 reconstruction_loss: ReconstructionLossBase,
                 attention_loss: AttentionLossBase,
                 distillation_loss: DistillationLossBase,
                 original_model: Union[OriginalModel, OriginalDataParallel],
                 reconstructed_model: Union[ReconstructedModel, ReconstructedDataParallel],
                 original_task_eval_fn: EvalFunction,
                 logger: Union[Logger, SummaryWriter],
                 task_dataloaders: Tuple[DataLoader, DataLoader],
                 device):
        self.config = config
        self.predictor = predictor
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

        self.exp_dir_path = create_experiment_dir(self.config.logging.log_dir, self.config.logging.exp_name)

        original_weights = self.original_model.get_learnable_weights()

        training_step = 0
        epoch = 0

        while True:
            for batch, ground_truth in self.train_dataloader:
                batch, ground_truth = batch.to(self.device), ground_truth.to(self.device)
                optimizer.zero_grad()

                reconstructed_weights = NeRNPredictorBase.predict_all(self.predictor, positional_embeddings,
                                                                      original_weights, learnable_weights_shapes)
                reconstructed_weights = self.reconstructed_model.sample_weights_by_shapes(reconstructed_weights)
                self.reconstructed_model.update_weights(reconstructed_weights)

                updated_weights = self.reconstructed_model.get_learnable_weights()

                original_outputs, original_feature_maps = self.original_model.get_feature_maps(batch)
                reconstructed_outputs, reconstructed_feature_maps = self.reconstructed_model.get_feature_maps(batch)

                reconstruction_loss = self.reconstruction_loss(updated_weights, original_weights)
                reconstruction_term = self.config.nern.reconstruction_loss_weight * reconstruction_loss

                attention_loss = self.attention_loss(reconstructed_feature_maps, original_feature_maps)
                distillation_loss = self.distillation_loss(reconstructed_outputs, original_outputs)

                if self._loss_warmup(training_step):
                    attention_term, distillation_term = torch.tensor(0), torch.tensor(0)
                else:
                    attention_term = self.config.nern.attention_loss_weight * attention_loss \
                        if self.config.nern.attention_loss_weight > 0 else torch.tensor(0)
                    distillation_term = self.config.nern.distillation_loss_weight * distillation_loss \
                        if self.config.nern.distillation_loss_weight > 0 else torch.tensor(0)

                loss = reconstruction_term + attention_term + distillation_term
                if loss.isnan().item() is True and self.config.task.use_random_data is True:
                    # This can result in an infinite loop, be careful
                    print("Loss is NaN when using random data. Skipping this batch.")
                    continue

                for updated_weight in updated_weights:
                    updated_weight.grad = None

                loss.backward()
                torch.autograd.backward(reconstructed_weights, [w.grad for w in updated_weights])

                if training_step % self.config.logging.log_interval == 0 and not self.config.logging.disable_logging:
                    loss_dict = dict(loss=loss,
                                     reconstruction_loss=reconstruction_term,
                                     attention_loss=attention_term,
                                     distillation_loss=distillation_term)
                    self._log_training(training_step, updated_weights, loss_dict, scheduler.get_last_lr(),
                                       self.config.logging.verbose)
                if training_step % self.config.eval_loss_window_interval == 0:
                    self._add_to_loss_window((attention_loss + distillation_loss).item())
                    if len(self.loss_window) == self.config.eval_loss_window_size and (attention_loss + distillation_loss).item() <= min(self.loss_window):
                        self._eval(training_step, "greedy")

                self._clip_grad_norm()
                optimizer.step()
                scheduler.check_and_step(training_step)
                training_step += 1

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

    def _eval(self, iteration, log_suffix=""):
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
        self.predictor.save(os.path.join(self.exp_dir_path,
                                                f"nern_{self.config.logging.exp_name}_{checkpoint_suffix}.pth"))

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

        print(f"[{training_step}/{self.config.num_iterations}] Loss = {loss_dict['loss'].item():.8f} ({''.join(f'{k} = {v.item():.8f}, ' for k, v in loss_dict.items() if k != 'loss')})")
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
        predictor_grad_norms = compute_grad_norms(list(self.predictor.parameters()))
        log_scalar_list(predictor_grad_norms,
                        title='predictor_grad_norms',
                        series_name='layer',
                        iteration=training_step,
                        logger=self.logger)

    def _set_grads_for_training(self):
        self.original_model.eval()
        self.reconstructed_model.eval()
        for param in self.original_model.parameters():
            param.requires_grad = False
        for param in self.reconstructed_model.parameters():
            param.requires_grad = True

        self.predictor.train()
        for param in self.predictor.parameters():
            param.requires_grad = True

    def _initialize_optimizer_and_scheduler(self) -> Tuple[torch.optim.Optimizer, GenericScheduler]:
        parameters_for_optimizer = list(self.predictor.parameters())
        if self.config.learn_fc_layer is True:
            parameters_for_optimizer.append(self.reconstructed_model.get_fully_connected_weights())

        optimizer = OptimizerFactory.get(parameters_for_optimizer, self.config)
        scheduler = LRSchedulerFactory.get(optimizer, self.config)

        return optimizer, scheduler

    def _set_training_steps(self):
        if self.config.num_iterations is None:
            self.config.num_iterations = self.config.epochs * len(self.train_dataloader)
