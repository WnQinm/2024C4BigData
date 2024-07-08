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
                                               batch_first=True)

    def forward(self, x:torch.Tensor):
        # input&output (batch_size, seq_len, feature_dim)
        x = torch.fft.fft(x, dim=1)
        x_real = torch.real(x)
        x_imag = torch.imag(x)
        x_real, _ = self.att_layer(x_real, x_real, x_real)
        x_imag, _ = self.att_layer(x_imag, x_imag, x_imag)
        return torch.fft.ifft(x_real + 1j * x_imag, dim=1)

