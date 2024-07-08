import torch
import torch.nn as nn
from models.FAt import FAt
from models.iTransformer import iTransformer


class Model(nn.Module):
    def __init__(self, args) -> None:
        super(Model, self).__init__()
        self.batch_size = args.batch_size
        self.lstm_layer_num = args.lstm_layer_num
        self.hidden_size = args.lstm_hidden_size
        self.pred_len = args.pred_len

        self.iTrm = iTransformer(args)
        self.FAt = FAt(args)
        self.output_dim = args.feature_len+args.fat_feature_dim
        self.fc1 = nn.Linear(self.output_dim, self.output_dim, dtype=float)
        self.gelu = nn.GELU()
        self.lstm = nn.LSTM(input_size=self.output_dim,
                            hidden_size=args.lstm_hidden_size,
                            num_layers=args.lstm_layer_num,
                            batch_first=True,
                            dropout=args.dropout)
        self.fc2 = nn.Linear(args.lstm_hidden_size, self.output_dim, dtype=float)
        self.fc3 = nn.Linear(self.output_dim, args.label_len, dtype=float)

    def forward(self, x:torch.Tensor):
        a = self.iTrm(x)
        b = self.FAt(x).to(torch.float32)
        x = torch.cat((a, b), dim=-1)
        x = self.fc1(x)
        x = self.gelu(x)

        result = []
        # o (batch_size, seq_len, hidden_size)
        o, h = self.lstm(x)
        o = self.fc2(o[:, -1:, :])
        result.append(o)
        for _ in range(self.pred_len-1):
            o, h = self.lstm(o, h)
            o = self.fc2(o)
            result.append(o)
        x = torch.cat(result, dim=1)      # (batch_size, pred_len, hidden_size)
        x = self.gelu(x)
        x = self.fc3(x)
        return x