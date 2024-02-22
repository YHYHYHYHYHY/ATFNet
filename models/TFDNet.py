import torch.nn.functional as F
from torch import nn
import torch
import argparse
import numpy as np
from layers.Autoformer_EncDec import series_decomp

class SeasonalEncoder(nn.Module):
    def __init__(self, configs):
        super(SeasonalEncoder, self).__init__()
        self.n_fft = configs.seq_len // 8
        self.n_frame = configs.seq_len // (self.n_fft // 4) + 1
        self.d_spec = self.n_fft // 2 + 1
        self.TFB_r = nn.Parameter(torch.randn(configs.enc_in, self.n_frame, self.d_spec, self.d_spec))
        self.TFB_i = nn.Parameter(torch.randn(configs.enc_in, self.n_frame, self.d_spec, self.d_spec))

    def comp_mul(self, x_r, x_i, y_r, y_i):
        r = torch.einsum('bdmn,dmnn->bdmn', x_r, y_r) - torch.einsum('bdmn,dmnn->bdmn', x_i, y_i)
        i = torch.einsum('bdmn,dmnn->bdmn', x_i, y_i) - torch.einsum('bdmn,dmnn->bdmn', x_r, y_r)
        return r + 1j * i
    def forward(self, x): # x.shape = [B, L, n_vars]
        B, L, n_vars = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, L)
        spec = torch.stft(x, n_fft=self.n_fft, return_complex=True).permute(0, 2, 1)
        spec = spec.view(B, n_vars, self.n_frame, -1)
        q = self.comp_mul(spec.real, spec.imag, self.TFB_r, self.TFB_i)
        z = q + spec
        z = torch.flatten(z, 0, 1)
        out = torch.istft(z.permute(0, 2, 1), n_fft=self.n_fft)
        return out.view(B, n_vars, -1)

class TrendEncoder(nn.Module):
    def __init__(self, configs):
        super(TrendEncoder, self).__init__()
        self.n_fft = configs.seq_len // 8
        self.n_frame = configs.seq_len // (self.n_fft // 4) + 1
        self.d_spec = self.n_fft // 2 + 1
        self.TFB_r = nn.Parameter(torch.randn(self.n_frame, self.d_spec, self.d_spec))
        self.TFB_i = nn.Parameter(torch.randn(self.n_frame, self.d_spec, self.d_spec))

    def comp_mul(self, x_r, x_i, y_r, y_i):
        r = torch.einsum('bdmn,mnn->bdmn', x_r, y_r) - torch.einsum('bdmn,mnn->bdmn', x_i, y_i)
        i = torch.einsum('bdmn,mnn->bdmn', x_i, y_i) - torch.einsum('bdmn,mnn->bdmn', x_r, y_r)
        return r + 1j * i
    def forward(self, x): # x.shape = [B, L, n_vars]
        B, L, n_vars = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, L)
        spec = torch.stft(x, n_fft=self.n_fft, return_complex=True).permute(0, 2, 1)
        spec = spec.view(B, n_vars, self.n_frame, -1)
        q = self.comp_mul(spec.real, spec.imag, self.TFB_r, self.TFB_i)
        z = q + spec
        z = torch.flatten(z, 0, 1)
        out = torch.istft(z.permute(0, 2, 1), n_fft=self.n_fft)
        return out.view(B, n_vars, -1)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.decomp = series_decomp(kernel_size=25)
        self.seasonal_encoder = SeasonalEncoder(configs)
        self.trend_encoder = TrendEncoder(configs)
        self.projection = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec): # [B, L, n_vars]
        means = torch.mean(x_enc, dim=1, keepdim=True)
        std = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True) + 1e-5)
        x_enc = (x_enc - means) / std

        seasonal, trend = self.decomp(x_enc)

        seasonal_out = self.seasonal_encoder(seasonal)
        trend_out = self.trend_encoder(trend)

        out = seasonal_out + trend_out
        out = self.projection(out).permute(0, 2, 1)

        out = out * std + means

        return out





