import torch
import torch.nn as nn
from layers.Pyraformer_EncDec import Encoder


class Model(nn.Module):
    """
    Pyraformer: Pyramidal attention to reduce complexity
    Paper link: https://openreview.net/pdf?id=0EXmFzUn5I
    """

    def __init__(self, configs, window_size=[4,4], inner_size=5):
        """
        window_size: list, the downsample window size in pyramidal attention.
        inner_size: int, the size of neighbour attention
        """
        super().__init__()
        self.task_name = "long_term_forecast"
        self.task = configs.train_type
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model

        if self.task_name == 'short_term_forecast':
            window_size = [2,2]
        self.encoder = Encoder(configs, window_size, inner_size)

        self.projection = nn.Linear(
            (len(window_size)+1)*self.d_model, self.pred_len * configs.enc_in)

    def long_forecast(self, x_enc, x_mark_enc):
        enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]
        dec_out = self.projection(enc_out).view(
            enc_out.size(0), self.pred_len, -1)
        return dec_out

    def short_forecast(self, x_enc, x_mark_enc):
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]
        dec_out = self.projection(enc_out).view(
            enc_out.size(0), self.pred_len, -1)

        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def forward(self, x_enc, x_mark_enc=None):
        if self.task_name in ['long_term_forecast', 'global_forecast']:
            dec_out = self.long_forecast(x_enc, x_mark_enc)
        elif self.task_name == 'short_term_forecast':
            dec_out = self.short_forecast(x_enc, x_mark_enc)
        else:
            raise NotImplementedError
        
        dec_out = dec_out[:, -self.pred_len:, :]
        if self.task == "wind":
            return dec_out[..., -2:-1]
        elif self.task == "temp":
            return dec_out[..., -1:]
        else:
            return None