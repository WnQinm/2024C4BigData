from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import adjust_learning_rate
from utils.loss import MSELoss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self):
        data_set, data_loader = data_provider(self.args)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self) -> MSELoss:
        criterion = MSELoss()
        return criterion

    def train(self, setting):
        _, train_loader = self._get_data()

        save_path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            train_loss = []

            self.model.train()

            pbar = tqdm(range(train_steps))
            pbar.set_description(f'Epoch {epoch}')

            best_loss = 100

            # (batch_size, seq_len, 4*9+2) (batch_size, pred_len, 2)
            for i, (batch_x, batch_y) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # outputs (batch_size, pred_len, 38)
                        if self.args.output_attention:
                            outputs = self.model(batch_x)[0]
                        else:
                            outputs = self.model(batch_x)

                        # TODO 这里先直接用后两位做预测(对应dataset后两位是wind和temp)，大概率需要再做处理
                        f_dim = -1 if self.args.features == 'MS' else -2
                        outputs = outputs[:, :, f_dim:]
                        batch_y = batch_y[:, :, f_dim:]
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    # outputs (batch_size, pred_len, 38)
                    if self.args.output_attention:
                        outputs = self.model(batch_x)[0]
                    else:
                        outputs = self.model(batch_x)

                    # TODO 这里先直接用后两位做预测(对应dataset后两位是wind和temp)，大概率需要再做处理
                    f_dim = -1 if self.args.features == 'MS' else -2
                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    pbar.set_postfix(loss=loss.item())

                if loss.item()<best_loss:
                    best_loss = loss.item()
                    torch.save(self.model.state_dict(), save_path + '/' + f'checkpoint_{epoch}_best.pth')

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                pbar.update(1)

            train_loss = np.average(train_loss)
            print(f"epoch average loss: {train_loss}")
            torch.save(self.model.state_dict(), save_path + '/' + f'checkpoint_{epoch}_last.pth')
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        return self.model