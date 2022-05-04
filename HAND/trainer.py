from torch import optim

from HAND.predictors.predictor import HANDPredictorBase
from HAND.eval_func import EvalFunction
from HAND.tasks.mnist.mnist_train_main import get_dataloaders
from logger import create_experiment_dir
from loss import ReconstructionLoss, FeatureMapsDistillationLoss, OutputDistillationLoss
from HAND.models.model import OriginalModel, ReconstructedModel
from options import TrainConfig
from tqdm import tqdm


class Trainer:
    def __init__(self,
                 config: TrainConfig,
                 predictor: HANDPredictorBase,
                 reconstruction_loss: ReconstructionLoss,
                 feature_maps_distillation_loss: FeatureMapsDistillationLoss,
                 output_distillation_loss: OutputDistillationLoss,
                 original_model: OriginalModel,
                 reconstructed_model: ReconstructedModel,
                 original_task_eval_fn: EvalFunction):
        self.config = config
        self.predictor = predictor
        self.reconstruction_loss = reconstruction_loss
        self.feature_maps_distillation_loss = feature_maps_distillation_loss
        self.output_distillation_loss = output_distillation_loss
        self.original_model = original_model
        self.reconstructed_model = reconstructed_model
        self.original_task_eval_fn = original_task_eval_fn

    def train(self):
        self._set_grads_for_training()

        exp_dir = create_experiment_dir(self.config.log_dir, self.config.exp_name)
        optimizer = self._initialize_optimizer()

        data_kwargs = {'batch_size': self.config.batch_size}
        test_dataloader, train_dataloader = get_dataloaders(test_kwargs=data_kwargs,
                                                            train_kwargs=data_kwargs)

        # For a number of epochs
        for epoch in tqdm(range(self.config.epochs), desc='epochs'):
            indices, positional_embeddings = self.reconstructed_model.get_positional_embeddings()
            for index, positional_embedding in zip(indices, positional_embeddings):
                # Reconstruct all of the original model's weights using the predictors model
                reconstructed_weights = self.predictor(positional_embedding)
                self.reconstructed_model.update_weights(index, reconstructed_weights)

            # Now we can see how good our reconstructed model is
            reconstruction_term = self.config.hand.reconstruction_loss_weight * self.reconstruction_loss(
                self.reconstructed_model, self.original_model)
            # feature_maps_term = self.config.hand.feature_maps_distillation_loss_weight * self.feature_maps_distillation_loss(
            #     batch, self.reconstructed_model, self.original_model)  # TODO: where does the batch come from? which loop
            # outputs_term = self.config.hand.output_distillation_loss_weight * self.output_distillation_loss(
            #     self.reconstructed_model, self.original_model)# TODO: where does the batch come from? which loop
            # loss = reconstruction_term + feature_maps_term + outputs_term
            loss = reconstruction_term
            loss.backward()
            optimizer.step()
            if epoch % self.config.eval_epochs_interval == 0:
                self.original_task_eval_fn.eval(self.reconstructed_model, test_dataloader)

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
        optimizer = optimizer_type(self.predictor.parameters(),
                                   betas=self.config.betas)
        return optimizer
