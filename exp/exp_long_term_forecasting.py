from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import adjust_learning_rate
from utils.loss import MSELoss
import torch
import torch.nn as nn
from torch import optim
import os
from tqdm import tqdm
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.writer = None
        if args.tensorboard is not None and args.tensorboard != "None":
            from torch.utils.tensorboard import SummaryWriter
            if not os.path.exists(args.tensorboard):
                os.makedirs(args.tensorboard)
            self.writer = SummaryWriter(args.tensorboard)

    def get_parameter_number(self, model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_num, trainable_num

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.ckpt_path is not None and self.args.ckpt_path != "None":
            pretrained_dict = torch.load(self.args.ckpt_path)
            model.load_state_dict(pretrained_dict)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        param_num = self.get_parameter_number(model)
        print(f"parameter total {param_num[0]} trainable {param_num[1]}")
        return model

    def _get_data(self):
        data_set, data_loader = data_provider(self.args)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = MSELoss()
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data()
        train_steps = len(train_loader)

        save_path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        best_loss = None

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            pbar = tqdm(range(train_steps))
            self.writer.add_scalar(f'Progress/epoch', (epoch+1)/self.args.train_epochs, epoch)

            self.model.train()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x)[0]
                        else:
                            outputs = self.model(batch_x)

                        if self.args.task == 'wind':
                            outputs = outputs[:, :, -2:-1]
                        elif self.args.task == 'temp':
                            outputs = outputs[:, :, -1:]
                        elif self.args.task == 'both':
                            outputs = outputs[:, :, -2:]

                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x)[0]
                    else:
                        outputs = self.model(batch_x)

                    if self.args.task == 'wind':
                        outputs = outputs[:, :, -2:-1]
                    elif self.args.task == 'temp':
                        outputs = outputs[:, :, -1:]
                    elif self.args.task == 'both':
                        outputs = outputs[:, :, -2:]

                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if self.writer is not None:
                    self.writer.add_scalar(f'epoch{epoch}/Loss/train', train_loss[-1], i)
                    self.writer.add_scalar(f'Progress/lr', model_optim.param_groups[0]['lr'], i)
                    self.writer.add_scalar(f'Progress/step', (i+1)/train_steps, i)

                if (i + 1) % 10 == 0:
                    mean_loss = np.mean(train_loss[-10:])
                    if best_loss is None or mean_loss<best_loss:
                        best_loss = mean_loss
                        torch.save(self.model.state_dict(), save_path + '/' + f'checkpoint_best.pth')

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                pbar.update(1)

            torch.save(self.model.state_dict(), save_path + '/' + f'checkpoint.pth')
            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        return self.model