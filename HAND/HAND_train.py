import json
import os

import pyrallis
import torch

from HAND.eval_func import EvalFunction
from HAND.logger import initialize_clearml_task
from HAND.tasks.vgg8 import ReconstructedVGG83x3, VGG8
from HAND.loss.attention_loss import AttentionLossFactory
from HAND.loss.reconstruction_loss import ReconstructionLossFactory
from HAND.loss.distillation_loss import DistillationLossFactory
from HAND.loss.task_loss import TaskLossFactory
from HAND.options import TrainConfig
from HAND.predictors.predictor import HANDPredictorFactory
from HAND.trainer import Trainer
from HAND.tasks.dataloader_factory import DataloaderFactory
from HAND.tasks.model_factory import ModelFactory


def load_original_model(cfg: TrainConfig, device: torch.device):
    model_kwargs_path = cfg.original_model_path.replace('pt', 'json')
    if os.path.exists(model_kwargs_path):
        with open(model_kwargs_path) as f:
            model_kwargs = json.load(f)
    else:
        model_kwargs = dict()
    original_model, reconstructed_model = ModelFactory.get(cfg, device, **model_kwargs)
    return original_model, reconstructed_model


@pyrallis.wrap()
def main(cfg: TrainConfig):
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    original_model, reconstructed_model = load_original_model(cfg, device)

    pos_embedding = reconstructed_model.positional_encoder.output_size
    predictor = HANDPredictorFactory(cfg.hand, input_size=pos_embedding).get_predictor().to(device)

    for p in predictor.parameters():
        if len(p.shape) >= 2:
            p.data = torch.fmod(p.data, 2)

    num_predictor_params = sum([p.numel() for p in predictor.parameters()])
    print(f"Predictor:"
          f"\t-> Number of parameters: {num_predictor_params / 1000}K"
          f"\t-> Size: {num_predictor_params * 4 / 1024 / 1024:.2f}Mb")

    num_predicted_params = sum([p.numel() for p in original_model.get_learnable_weights()])
    print(f"\nOriginal Model:"
          f"\t-> Number of parameters: {num_predicted_params / 1000}K"
          f"\t-> Size: {num_predicted_params * 4 / 1024 / 1024:.2f}Mb")

    if not cfg.logging.disable_logging:
        clearml_task = initialize_clearml_task(cfg.logging.exp_name)
        clearml_logger = clearml_task.get_logger()
    else:
        clearml_logger = None

    dataloaders = DataloaderFactory.get(cfg.task.task_name, **{'batch_size': cfg.batch_size})

    trainer = Trainer(config=cfg,
                      predictor=predictor,
                      task_loss=TaskLossFactory.get(cfg.hand.task_loss_type),
                      reconstruction_loss=ReconstructionLossFactory.get(cfg.hand.reconstruction_loss_type),
                      attention_loss=AttentionLossFactory.get(cfg.hand.attention_loss_type),
                      distillation_loss=DistillationLossFactory.get(cfg.hand.distillation_loss_type),
                      original_model=original_model,
                      reconstructed_model=reconstructed_model,
                      original_task_eval_fn=EvalFunction(),
                      logger=clearml_logger,
                      task_dataloaders=dataloaders,
                      device=device)
    trainer.train()


if __name__ == '__main__':
    main()
