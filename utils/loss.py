import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        MSE_wind = F.mse_loss(output[..., 0], target[..., 0], reduction="mean")
        MSE_temp = F.mse_loss(output[..., 1], target[..., 1], reduction="mean")
        return MSE_wind + 10 * MSE_temp