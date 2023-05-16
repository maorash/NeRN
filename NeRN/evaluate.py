import pyrallis
import torch

from NeRN.options import Config
from NeRN.predictors.factory import NeRNPredictorFactory
from NeRN.tasks.model_factory import load_original_model
from NeRN.predictors.predictor import NeRNPredictorBase
from NeRN.eval_func import EvalFunction
from NeRN.tasks.dataloader_factory import DataloaderFactory


@pyrallis.wrap()
def main(cfg: Config):
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    original_model, reconstructed_model = load_original_model(cfg, device)

    predictor = NeRNPredictorFactory(cfg, input_size=reconstructed_model.output_size).get_predictor().to(device)

    init_predictor(cfg, predictor)

    learnable_weights_shapes = reconstructed_model.get_learnable_weights_shapes()
    indices, positional_embeddings = reconstructed_model.get_indices_and_positional_embeddings()
    reconstructed_weights = NeRNPredictorBase.predict_all(predictor,
                                                          positional_embeddings,
                                                          original_model.get_learnable_weights(),
                                                          learnable_weights_shapes)
    reconstructed_weights = reconstructed_model.sample_weights_by_shapes(reconstructed_weights)
    reconstructed_model.update_weights(reconstructed_weights)

    _, test_dataloader = DataloaderFactory.get(cfg.task, **{'batch_size': cfg.batch_size,
                                                            'num_workers': cfg.num_workers,
                                                            'only_test': True})

    eval_func = EvalFunction(cfg)
    eval_func.eval(reconstructed_model, test_dataloader, iteration=0, logger=None)


def init_predictor(cfg, predictor):
    if cfg.nern.init != "checkpoint":
        raise ValueError("Please pass a checkpoint to init the predictor")
    print(f"Loading pretrained weights from: {cfg.nern.checkpoint_path}")
    predictor.load(cfg.nern.checkpoint_path)


if __name__ == '__main__':
    main()
