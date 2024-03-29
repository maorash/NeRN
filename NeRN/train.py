import json
import os

import pyrallis
import torch
from torch.utils.tensorboard import SummaryWriter
from clearml import Task

import NeRN.log_utils as log_utils
from NeRN.loss.attention_loss import AttentionLossFactory
from NeRN.loss.distillation_loss import DistillationLossFactory
from NeRN.loss.reconstruction_loss import ReconstructionLossFactory
from NeRN.options import Config
from NeRN.predictors.factory import NeRNPredictorFactory
from NeRN.tasks.model_factory import load_original_model
from NeRN.trainer import Trainer
from NeRN.eval_func import EvalFunction
from NeRN.tasks.dataloader_factory import DataloaderFactory


@pyrallis.wrap()
def main(cfg: Config):
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    original_model, reconstructed_model = load_original_model(cfg, device)

    predictor = NeRNPredictorFactory(cfg, input_size=reconstructed_model.output_size).get_predictor().to(device)

    init_predictor(cfg, predictor)

    if not cfg.logging.disable_logging:
        if cfg.logging.use_tensorboard:
            logger = SummaryWriter(log_dir=os.path.join(cfg.logging.log_dir, "tb_logs", cfg.logging.exp_name))
            logger.add_text("Config", json.dumps(pyrallis.encode(cfg), indent=4))
        else:
            clearml_task = Task.init(project_name='NeRN', task_name=cfg.logging.exp_name, deferred_init=True)
            clearml_task.connect(log_utils.flatten(pyrallis.encode(cfg)))  # Flatten because of clearml bug
            logger = clearml_task.get_logger()
    else:
        logger = None

    num_predictor_params = sum([p.numel() for p in predictor.parameters()])
    print(f"Predictor:"
          f"\t-> Number of parameters: {num_predictor_params / 1000}K"
          f"\t-> Size: {num_predictor_params * 4 / 1024 / 1024:.2f}Mb")

    num_predicted_params = sum([p.numel() for p in original_model.get_learnable_weights()])
    num_total_params = sum([p.numel() for p in original_model.parameters()])
    print(f"\nOriginal Model:"
          f"\t-> Number of learnable parameters: {num_predicted_params / 1000}K"
          f"\t-> Size of learnable parameters: {num_predicted_params * 4 / 1024 / 1024:.2f}Mb",
          f"\n\t-> Total model size: {num_total_params * 4 / 1024 / 1024:.2f}Mb")

    dataloaders = DataloaderFactory.get(cfg.task, **{'batch_size': cfg.batch_size,
                                                     'num_workers': cfg.num_workers})

    trainer = Trainer(config=cfg,
                      predictor=predictor,
                      reconstruction_loss=ReconstructionLossFactory.get(cfg.nern),
                      attention_loss=AttentionLossFactory.get(cfg.nern),
                      distillation_loss=DistillationLossFactory.get(cfg.nern),
                      original_model=original_model,
                      reconstructed_model=reconstructed_model,
                      original_task_eval_fn=EvalFunction(cfg),
                      logger=logger,
                      task_dataloaders=dataloaders,
                      device=device)
    trainer.train()


def init_predictor(cfg, predictor):
    if cfg.nern.init == "fmod":
        print("Initializing using fmod")
        for p in predictor.parameters():
            if len(p.shape) >= 2:
                p.data = torch.fmod(p.data, 2)
    elif cfg.nern.init == "checkpoint":
        print(f"Loading pretrained weights from: {cfg.nern.checkpoint_path}")
        predictor.load(cfg.nern.checkpoint_path)
    elif cfg.nern.init == "default":
        print("Using default torch initialization")
    else:
        raise ValueError(f"Unsupported initialization method: {cfg.nern.init}")


if __name__ == '__main__':
    main()
