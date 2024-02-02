import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from torch.nn.functional import relu



class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, N, L, H, E = queries.shape
        _, N, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("bnlhe,bnshe->bnhls", queries, keys)
        A = torch.softmax(scale * scores, dim=-1)
        # A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bnhls,bnshd->bnlhd", A, values)
        return V.contiguous()



class FreqAttentionLayer(nn.Module):
    def __init__(self,  d_model, n_heads, attention=FullAttention(), d_keys=None,
                 d_values=None):
        super(FreqAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model * 2, d_keys * n_heads * 2)
        self.key_projection = nn.Linear(d_model * 2, d_keys * n_heads * 2)
        self.value_projection = nn.Linear(d_model * 2, d_values * n_heads * 2)
        self.out_projection = nn.Linear(d_values * n_heads * 2, d_model * 2)
        self.n_heads = n_heads

    def forward(self, queries, keys, values): # queries.shape = [batch_size, n_vars, d_model, 2]

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, -1, H, 2)
        keys = self.key_projection(keys).view(B, S, -1, H, 2)
        values = self.value_projection(values).view(B, S, -1, H, 2)

        out= self.inner_attention(
            queries,
            keys,
            values,
        )
        out = out.view(B, L, -1)

        return self.out_projection(out)

class FreqEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout=0.2):
        super(FreqEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model * 2)
        self.norm2 = nn.LayerNorm(d_model * 2)
        self.attention = attention
        self.Linear1 = nn.Linear(d_model * 2, d_ff * 2)
        self.Linear2 = nn.Linear(d_ff * 2, d_model * 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x): # x.shape = [batch_size, n_vars, d_model, 2]
        x = x.view(x.shape[0], x.shape[1], -1)
        new_x = self.attention(
            x, x, x
        )
        new_x = self.dropout(new_x)
        x = x + new_x

        x = self.norm1(x)
        y = x
        y = self.relu(self.Linear1(y))
        y = self.dropout(y)
        y = self.Linear2(y)
        y = self.dropout(y)

        out = self.norm2(x + y)
        out = out.view(out.shape[0], out.shape[1], -1, 2)
        return out

class FreqEncoder(nn.Module):
    def __init__(self, attn_layers):
        super(FreqEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
    def forward(self, x):
        for attn_layer in self.attn_layers:
            x = attn_layer(x)
        return x

class FreqEncoderBlock(nn.Module):
    
    def __init__(self, configs):
        super(FreqEncoderBlock, self).__init__()
        self.configs = configs
        self.device = configs.device


        self.ori_len = int((configs.seq_len) / 2) + 1
        self.tar_len = int((configs.seq_len + configs.pred_len) / 2) + 1
        self.d_ff = configs.fnet_d_ff
        self.d_model = configs.fnet_d_model
        self.emb = nn.Linear(self.ori_len * 2, self.d_model * 2)
        self.dropout = nn.Dropout(configs.complex_dropout)
        self.projection = nn.Linear(self.d_model * 2, self.tar_len * 2)
        self.encoder = FreqEncoder(
            attn_layers=[
                FreqEncoderLayer(
                    attention=FreqAttentionLayer(
                        d_model=self.d_model, n_heads=configs.n_heads
                    ),
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    dropout=configs.complex_dropout
                ) for _ in range(configs.e_layers)
            ]
        )

    def forward(self, x):  # x.shape: [B, L:seq_len/2, n_vars, 2]
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(x.shape[0], x.shape[1], -1)
        x = self.emb(x).view(x.shape[0], x.shape[1], -1, 2)
        x = self.dropout(x)
        x = self.encoder(x) # x.shape: [B, n_vars, d_model, 2]
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.projection(x).view(x.shape[0], x.shape[1], -1, 2)
        return x.permute(0, 2, 1, 3)