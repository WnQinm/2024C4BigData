import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """
    def __init__(self, configs):
        super(iTransformer, self).__init__()
        self.seq_len = configs.seq_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.itrm_e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        self.projection = nn.Linear(configs.d_model, configs.seq_len, dtype=float)

    def forecast(self, x_enc: torch.Tensor):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        enc_out = self.enc_embedding(x_enc, 1)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)      # (batch_size, feature_len, d_model)
        dec_out = self.projection(enc_out).permute(0, 2, 1)         # (batch_size, seq_len, feature_len)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def forward(self, x_enc):
        # (batch_size, seq_len, feature_len)
        dec_out = self.forecast(x_enc)
        return dec_out