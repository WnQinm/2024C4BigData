import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def masked_mse_loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        mask = (y_true != 0).float()
        mask /= mask.mean()
        loss = torch.pow(y_true - y_pred, 2)
        loss = loss * mask
        # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
        loss[loss != loss] = 0
        return loss.mean()

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        MSE_wind = F.mse_loss(output[..., 0], target[..., 0], reduction="mean")
        MSE_temp = F.mse_loss(output[..., 1], target[..., 1], reduction="mean")
        return MSE_wind + 10 * MSE_temp