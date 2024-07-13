from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import adjust_learning_rate
from utils.loss import MSELoss
from models.Pyraformer import Model
import torch
import torch.nn as nn
from torch import optim
import os
import warnings
import numpy as np
from tqdm import tqdm

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
        model = Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        param_num = self.get_parameter_number(model)
        print(f"parameter total {param_num[0]} trainable {param_num[1]}")
        return model

    def _get_data(self):
        return data_provider(self.args)

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self) -> MSELoss:
        criterion = MSELoss()
        return criterion

    # TODO 后处理(平滑之类的)
    def _get_loss(self, pred, label, criterion):
        assert pred.shape[-1]==self.args.label_len and label.shape[-1]==self.args.label_len
        loss = criterion(pred, label)
        return loss

    def train(self, setting):
        train_loader, eval_loader = self._get_data()

        save_path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if self.args.ckpt_path is not None and self.args.ckpt_path != "None":
            pretrained_dict = torch.load(self.args.ckpt_path)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc2' not in k)}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        best_loss = None

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            train_loss = []
            eval_dataloader = iter(eval_loader)

            self.model.train()

            pbar = tqdm(range(train_steps))
            if self.writer is not None:
                self.writer.add_scalar(f'Progress/epoch', (epoch+1)/self.args.train_epochs, epoch)
            else:
                pbar.set_description(f'Epoch {epoch+1}/{self.args.train_epochs}')

            # (batch_size, seq_len, 4*9+2) (batch_size, pred_len, 1)
            for i, (batch_x, batch_y) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)

                        loss = self._get_loss(outputs, batch_y, criterion)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x)
                    loss = self._get_loss(outputs, batch_y, criterion)
                    train_loss.append(loss.item())

                if self.writer is not None:
                    self.writer.add_scalar(f'epoch{epoch}/Loss/train', loss, i)
                    self.writer.add_scalar(f'Progress/lr', model_optim.param_groups[0]['lr'], i)
                    self.writer.add_scalar(f'Progress/step', (i+1)/train_steps, i)

                if (i + 1) % self.args.eval_step == 0:
                    # evaluate
                    with torch.no_grad():
                        eval_batch_x, eval_batch_y = next(eval_dataloader)
                        eval_batch_x = eval_batch_x.float().to(self.device)
                        eval_batch_y = eval_batch_y.float().to(self.device)
                        eval_outputs = self.model(eval_batch_x)
                        eval_loss = self._get_loss(eval_outputs, eval_batch_y, criterion)
                    if best_loss is None or np.mean(train_loss[-self.args.eval_step:])<best_loss:
                        best_loss = np.mean(train_loss[-self.args.eval_step:])
                        torch.save(self.model.state_dict(), save_path + '/' + f'checkpoint_{epoch}_best.pth')

                    if self.writer is not None:
                        self.writer.add_scalar(f"epoch{epoch}/Loss/val", eval_loss, i)
                    else:
                        pbar.set_postfix(train_loss=loss.item(), eval_loss=eval_loss.item())

                if self.args.autosave is not None and (i + 1) % (train_steps * self.args.autosave //100) == 0:
                    torch.save(self.model.state_dict(), save_path + '/' + f'checkpoint_{epoch}_autosave.pth')

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                pbar.update(1)

            train_loss = np.average(train_loss)
            # print(f"epoch {epoch} average loss: {train_loss}")
            torch.save(self.model.state_dict(), save_path + '/' + f'checkpoint_{epoch}_last.pth')
            adjust_learning_rate(model_optim, epoch, self.args)

        return self.model