from torch import nn


class ActivationsFactory:
    activations = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "hardswish": nn.Hardswish,
        "gelu": nn.GELU,
        "relu6": nn.ReLU6,
        "elu": nn.ELU
    }

    @staticmethod
    def get(activation_type: str = "relu") -> nn.Module:
        try:
            return ActivationsFactory.activations[activation_type]()
        except KeyError:
            raise ValueError("Unknown activation type")