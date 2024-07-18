import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import mse_loss


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        a = mse_loss(input[..., -2], target[..., -2], reduction='mean')
        b = mse_loss(input[..., -1], target[..., -1], reduction='mean')
        return (a + b) / 2
