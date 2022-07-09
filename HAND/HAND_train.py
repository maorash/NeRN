import pyrallis
import torch

from HAND.eval_func import EvalFunction
from HAND.loss import ReconstructionLoss, FeatureMapsDistillationLoss, OutputDistillationLoss
from HAND.models.simple_net import SimpleNet, ReconstructedSimpleNet3x3
from HAND.options import TrainConfig
from HAND.predictors.predictor import HANDPredictorFactory
from HAND.trainer import Trainer


@pyrallis.wrap()
def main(cfg: TrainConfig):
    original_model = SimpleNet()  # TODO: factory and get this from config
    original_model.load_state_dict(torch.load('trained_models/original_tasks/mnist/mnist_cnn.pt'))

    reconstructed_model = ReconstructedSimpleNet3x3(original_model)

    predictor = HANDPredictorFactory(cfg.hand).get_predictor()

    trainer = Trainer(config=cfg, predictor=predictor,
                      reconstruction_loss=ReconstructionLoss(cfg.hand.reconstruction_loss_type),
                      feature_maps_distillation_loss=FeatureMapsDistillationLoss(
                          cfg.hand.feature_maps_distillation_loss_type),
                      output_distillation_loss=OutputDistillationLoss(cfg.hand.output_distillation_loss_type),
                      original_model=original_model,
                      reconstructed_model=reconstructed_model,
                      original_task_eval_fn=EvalFunction())
    trainer.train()


if __name__ == '__main__':
    main()
