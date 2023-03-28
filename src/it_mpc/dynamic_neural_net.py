# coding=utf-8
import pandas as pd

import os
import torch
from torch import nn
from torch.utils.data import DataLoader

# from multifeature_aggregator import (
#     aggregate_multiple_features,
#     StatePose2D,
#     CmdSkidSteer,
#     Velocity,
# )

from multifeature_aggregator.data_containers import AbstractMultifeature
from configure_dataset import marm1_eda, marm1_eda

# ::: Dynamic neural net :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# === IT-MPC dynamic net model =======================================================================
# Inspired from PyTorch doc: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

# Note: Mac M1 use `mps` instead of `cpu` <--
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class DynamicNetworkMLP(nn.Module):
    hidden_size: int
    obs_dim: int
    act_dim: int

    def __init__(self, hidden_size: int = 32, obs_dim: int = 6, act_dim: int = 4):
        super().__init__()

        self.hidden_size = hidden_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.flatten = nn.Flatten()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(self.obs_dim + self.act_dim, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.obs_dim),
        )

    def forward(self, state, action):
        input_ = self.flatten(state, action)
        logits_ = self.linear_tanh_stack(input_)
        return logits_


dynamic_network = DynamicNetworkMLP().to(device)
print(dynamic_network)

X = torch.rand(1, 6, device=device)
logits = dynamic_network(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
