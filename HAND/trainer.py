from torch import optim

from logger import create_experiment_dir
from predictor import HANDPredictorBase
from loss import ReconstructionLoss, DistillationLoss
from model import OriginalModel, ReconstructedModel
from options import TrainConfig


class Trainer:
    def __init__(self,
                 config: TrainConfig,
                 predictor: HANDPredictorBase,
                 reconstruction_loss: ReconstructionLoss,
                 distillation_loss: DistillationLoss,
                 original_model: OriginalModel,
                 reconstructed_model: ReconstructedModel):
        self.config = config
        self.predictor = predictor
        self.reconstruction_loss = reconstruction_loss
        self.distillation_loss = distillation_loss
        self.original_model = original_model
        self.reconstructed_model = reconstructed_model

    def train(self):
        exp_dir = create_experiment_dir(self.config.log_dir, self.config.exp_name)
        optimizer = self._initialize_optimizer()

        # For a number of epochs
        indices, positional_embeddings = self.reconstructed_model.get_positional_embeddings()
        for index, positional_embedding in zip(indices, positional_embeddings):
            # Reconstruct all of the original model's weights using the predictor model
            reconstructed_weights = self.predictor(positional_embedding)
            self.reconstructed_model.update_weights(index, reconstructed_weights)

        # Now we can see how good our reconstructed model is
        reconstruction_loss = self.reconstruction_loss(self.reconstructed_model, self.original_model)
        distillation_loss = self.distillation_loss(self.reconstructed_model, self.original_model)
        loss = self.config.hand.reconstruction_factor * reconstruction_loss + (
                1 - self.config.hand.reconstruction_factor) * distillation_loss
        loss.backward()
        optimizer.step()
