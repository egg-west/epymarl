import random
from typing import List

import numpy as np
import torch
import torch.nn as nn

ModelType = torch.nn.Module

def weight_init_linear(m: ModelType):
    assert isinstance(m.weight, TensorType)
    nn.init.xavier_uniform_(m.weight)
    assert isinstance(m.bias, TensorType)
    nn.init.zeros_(m.bias)


def weight_init_conv(m: ModelType):
    # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
    assert isinstance(m.weight, TensorType)
    assert m.weight.size(2) == m.weight.size(3)
    m.weight.data.fill_(0.0)
    if hasattr(m.bias, "data"):
        m.bias.data.fill_(0.0)  # type: ignore[operator]
    mid = m.weight.size(2) // 2
    gain = nn.init.calculate_gain("relu")
    assert isinstance(m.weight, TensorType)
    nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def weight_init_moe_layer(m: ModelType):
    assert isinstance(m.weight, TensorType)
    for i in range(m.weight.shape[0]):
        nn.init.xavier_uniform_(m.weight[i])
    assert isinstance(m.bias, TensorType)
    nn.init.zeros_(m.bias)


def weight_init(m: ModelType):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        weight_init_linear(m)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        weight_init_conv(m)
    else:
        raise NotImplementedError
    #elif isinstance(m, moe_layer.Linear):
    #    weight_init_moe_layer(m)

def _get_list_of_layers(
    input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
) -> List[nn.Module]:
    """Utility function to get a list of layers. This assumes all the hidden
    layers are using the same dimensionality.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): dimension of the hidden layers.
        output_dim (int): dimension of the output layer.
        num_layers (int): number of layers in the mlp.

    Returns:
        ModelType: [description]
    """
    mods: List[nn.Module]
    if num_layers == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        mods.append(nn.Linear(hidden_dim, output_dim))
    return mods

def build_mlp(
    input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
) -> ModelType:
    """Utility function to build a mlp model. This assumes all the hidden
    layers are using the same dimensionality.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): dimension of the hidden layers.
        output_dim (int): dimension of the output layer.
        num_layers (int): number of layers in the mlp.

    Returns:
        ModelType: [description]
    """
    mods: List[nn.Module] = _get_list_of_layers(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
    )
    return nn.Sequential(*mods)
