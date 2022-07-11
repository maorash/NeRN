import pyrallis
import torch

from HAND.eval_func import EvalFunction
from HAND.logger import initialize_clearml_task
from HAND.loss import TaskLoss, ReconstructionLoss, FeatureMapsDistillationLoss, OutputDistillationLoss
from HAND.models.simple_net import SimpleNet, ReconstructedSimpleNet3x3
from HAND.options import TrainConfig
from HAND.predictors.predictor import HANDPredictorFactory
from HAND.trainer import Trainer


@pyrallis.wrap()
def main(cfg: TrainConfig):
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    original_model = SimpleNet().to(device)  # TODO: factory and get this from config
    original_model.load_state_dict(torch.load('trained_models/original_tasks/mnist/mnist_cnn.pt'))

    reconstructed_model = ReconstructedSimpleNet3x3(original_model).to(device)

    predictor = HANDPredictorFactory(cfg.hand).get_predictor().to(device)

    clearml_task = initialize_clearml_task(cfg.logging.task_name)
    clearml_logger = clearml_task.get_logger()

    trainer = Trainer(config=cfg,
                      predictor=predictor,
                      task_loss=TaskLoss(cfg.hand.task_loss_type),
                      reconstruction_loss=ReconstructionLoss(cfg.hand.reconstruction_loss_type),
                      feature_maps_distillation_loss=FeatureMapsDistillationLoss(
                          cfg.hand.feature_maps_distillation_loss_type),
                      output_distillation_loss=OutputDistillationLoss(cfg.hand.output_distillation_loss_type),
                      original_model=original_model,
                      reconstructed_model=reconstructed_model,
                      original_task_eval_fn=EvalFunction(),
                      logger=clearml_logger,
                      device=device)
    trainer.train()


if __name__ == '__main__':
    main()
