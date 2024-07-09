import torch
import torch.nn as nn


class FAtLayer(nn.Module):
    '''
    Fourier Attention Layer
    '''
    def __init__(self, feature_dim, num_heads, dropout):
        super(FAtLayer, self).__init__()
        assert feature_dim%num_heads==0
        self.att_layer = nn.MultiheadAttention(embed_dim=feature_dim,
                                               num_heads=num_heads,
                                               batch_first=True,
                                               dropout=dropout)

    def forward(self, x:torch.Tensor):
        # input&output (batch_size, seq_len, feature_dim)
        x = torch.fft.rfft(x, dim=1)
        L = x.shape[1]
        x = torch.cat((torch.real(x), torch.imag(x)), dim=1)
        x, _ = self.att_layer(x, x, x)
        x = torch.fft.irfft(x[:, :L, :] + 1j * x[:, L:, :], dim=1)
        return x.to(torch.float32)
