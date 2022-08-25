import torch

from HAND.models.regularization import CosineSmoothness


def test_weights_regularization(weights):
    total_cosine = torch.zeros(1).to(weights[0].device)
    total_l2 = torch.zeros(1).to(weights[0].device)
    for layer_weight in weights:
        curr_cosine_reg, curr_l2_reg = CosineSmoothness.cosine_layer_smoothness(layer_weight)
        total_cosine += curr_cosine_reg
        total_l2 += curr_l2_reg
    print(total_cosine)
    print(total_l2)
