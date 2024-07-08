import torch
import torch.nn as nn
from layers.FAtLayer import FAtLayer
from layers.Embed import DataEmbedding


class FAt(nn.Module):

    def __init__(self, args):
        super(FAt, self).__init__()
        self.embedding = DataEmbedding(args.feature_len, args.fat_feature_dim, args.dropout)
        self.encoders = nn.ModuleList([FAtLayer(args.fat_feature_dim, args.n_heads, args.dropout)
                                      for _ in range(args.fat_e_layers)])

    def forward(self, x:torch.Tensor):
        # (batch_size, seq_len, feature_len)
        x = self.embedding(x, -1)
        for encoder in self.encoders:
            x = encoder(x)
        return  x
