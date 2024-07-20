import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import mse_loss, l1_loss


class MSELoss(nn.Module):
    def __init__(self, gradcum) -> None:
        super().__init__()
        self.gradcum = max(1, gradcum)

    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        # a = mse_loss(input[..., -2], target[..., -2], reduction='mean')
        # b = mse_loss(input[..., -1], target[..., -1], reduction='mean')
        # return (a + b) / (2 * max(1, self.gradcum))

        a1 = mse_loss(input[..., -2], target[..., -2], reduction='mean')
        a2 = l1_loss(input[..., -2], target[..., -2], reduction='mean')
        b1 = mse_loss(input[..., -1], target[..., -1], reduction='mean')
        b2 = l1_loss(input[..., -1], target[..., -1], reduction='mean')
        return (a1 + a2 + b1 + b2) / (4 * max(1, self.gradcum)), ((a1 + b1) / 2).item()
