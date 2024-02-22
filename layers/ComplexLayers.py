import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from torch.nn.functional import relu

def complex_mse(pred, label):
    delta = pred - label
    return torch.mean(torch.abs(delta) ** 2)

def complex_relu(input):
    return relu(input.real).type(torch.complex64) + 1j * relu(input.imag).type(torch.complex64)

def complex_dropout(input, dropout):
    mask_r = torch.ones(input.shape, dtype=torch.float32)
    mask_r = dropout(mask_r)
    mask_i = torch.zeros_like(mask_r)
    mask = torch.complex(mask_r, mask_i).to(input.device)
    return mask * input

def apply_complex(fr, fi, input, dtype=torch.complex64):
    return (fr(input.real) - fi(input.imag)).type(dtype) \
        + 1j * (fr(input.imag) + fi(input.real)).type(dtype)

def complex_mul(order, mat1, mat2):
    return (torch.einsum(order, mat1.real, mat2.real) - torch.einsum(order, mat1.imag, mat2.imag)) \
        + 1j * (torch.einsum(order, mat1.real, mat2.imag) - torch.einsum(order, mat1.imag, mat2.real))

class ComplexLN(nn.Module):
    def __init__(self, C):
        super(ComplexLN, self).__init__()
        z = torch.zeros((C))
        self.w_r = nn.Parameter(torch.tensor(sqrt(2)/2) + z)
        self.w_i = nn.Parameter(torch.tensor(sqrt(2)/2) + z)
        self.b_r = nn.Parameter(torch.tensor(0) + z)
        self.b_i = nn.Parameter(torch.tensor(0) + z)
    def forward(self, x):
        means = torch.mean(x, dim=-1, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=-1, keepdim=True))
        x = (x - means) / std
        w = torch.complex(self.w_r, self.w_i)
        b = torch.complex(self.b_r, self.b_i)
        return x * w + b

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)
    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)

class ComplexAttention(nn.Module):
    def __init__(self):
        super(ComplexAttention, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.is_activate = True
    def forward(self, q, k, v):
        scores = complex_mul("bnlhe,bnshe->bnhls", q, k)
        if self.is_activate:
            scores = self.dropout(torch.softmax(torch.abs(scores), dim=-1))
            scores = torch.complex(scores, torch.zeros_like(scores))
        V = complex_mul("bnhls,bnshd->bnlhd", scores, v)
        return V.contiguous()

class ComplexAttentionLayer(nn.Module):
    def __init__(self,  d_model, n_heads, attention=ComplexAttention(), d_keys=None,
                 d_values=None):
        super(ComplexAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = ComplexLinear(d_model, d_keys * n_heads)
        self.key_projection = ComplexLinear(d_model, d_keys * n_heads)
        self.value_projection = ComplexLinear(d_model, d_values * n_heads)
        self.out_projection = ComplexLinear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values): # queries.shape = [batch_size, n_vars, d_model]
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, -1, H, 1)
        keys = self.key_projection(keys).view(B, S, -1, H, 1)
        values = self.value_projection(values).view(B, S, -1, H, 1)

        out= self.inner_attention(
            queries,
            keys,
            values,
        )
        out = out.view(B, L, -1)

        return self.out_projection(out)

class ComplexEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout=0.2):
        super(ComplexEncoderLayer, self).__init__()
        self.norm1 = ComplexLN(d_model)
        self.norm2 = ComplexLN(d_model)
        self.attention = attention
        self.Linear1 = ComplexLinear(d_model, d_ff)
        self.Linear2 = ComplexLinear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        new_x = self.attention(
            x, x, x
        )
        new_x = complex_dropout(new_x, self.dropout)
        x = x + new_x

        x = self.norm1(x)
        y = x
        y = complex_relu(self.Linear1(y))
        y = complex_dropout(y, self.dropout)
        y = self.Linear2(y)
        y = complex_dropout(y, self.dropout)

        return self.norm2(x + y)

class ComplexEncoder(nn.Module):
    def __init__(self, attn_layers):
        super(ComplexEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
    def forward(self, x):
        for attn_layer in self.attn_layers:
            x = attn_layer(x)
        return x

class CompEncoderBlock(nn.Module):
    # Input: Extended_DFT(input sequence), shape: [B, L:(seq_len+pred_len)//2+1, n_vars] dtype: torch.cfloat
    # Output: DFT(input sequence + pred sequence), shape: [B, L:(seq_len+pred_len)//2+1, n_vars] dtype: torch.cfloat
    def __init__(self, configs, extended=True):
        super(CompEncoderBlock, self).__init__()
        self.configs = configs
        self.device = configs.device
        self.is_emb = True
        self.ori_len = int((configs.seq_len) / 2) + 1
        self.tar_len = int((configs.seq_len + configs.pred_len) / 2) + 1
        if extended:
            self.ori_len = int((configs.seq_len + configs.pred_len) / 2) + 1
        self.d_ff = configs.fnet_d_ff
        self.d_model = configs.fnet_d_model
        if not self.is_emb:
            self.d_model = self.tar_len
        self.emb = ComplexLinear(self.ori_len, self.d_model)
        self.dropout = nn.Dropout(configs.complex_dropout)
        self.projection = ComplexLinear(self.d_model, self.tar_len)
        self.encoder = ComplexEncoder(
            attn_layers=[
                ComplexEncoderLayer(
                    attention=ComplexAttentionLayer(
                        d_model=self.d_model, n_heads=configs.n_heads
                    ),
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    dropout = configs.complex_dropout
                ) for _ in range(configs.fnet_layers)
            ]
        )
    def forward(self, x):  
        x = x.permute(0, 2, 1)
        if self.is_emb:
            x = self.emb(x)
        x = complex_dropout(x, self.dropout)
        x = self.encoder(x) 
        x = self.projection(x)
        return x.permute(0, 2, 1)