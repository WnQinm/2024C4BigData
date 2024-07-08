import torch
import torch.nn as nn


class DataEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(DataEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x:torch.Tensor, dim:int):
        # 把指定维度移到最后
        index = list(range(len(x.shape)))
        dim = index[dim]
        index.remove(dim)
        index.append(dim)
        x = x.permute(*index)

        x = self.embedding(x)
        return self.dropout(x)
