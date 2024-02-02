import torch
import torch.nn as nn
import numpy as np
from layers.Autoformer_EncDec import series_decomp
from models import Autoformer, TimesNet, DLinear, FEDformer, \
    Informer, PatchTST, FITS, FreTS, FreqNet, CompNet, \
    CompNetv3, iTransformer, SCINet, Crossformer
from layers.ComplexLayers import CompEncoderBlock
from layers.Embed import PatchEmbedding
from models.PatchLinear import LinearLayer, FlattenHead
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer

class T_Block(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Prediction Head
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,head_dropout=configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out


class F_Block(nn.Module):
    def __init__(self, configs):
        super(F_Block, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.model = CompEncoderBlock(configs)


    def forward(self, x_enc, x_enc_mark, x_dec, x_dec_mark):
        paddings = torch.zeros((x_enc.shape[0], self.pred_len ,x_enc.shape[2])).to(x_enc.device)
        x_enc = torch.concatenate((x_enc, paddings), dim=1)
        freq = torch.fft.rfft(x_enc, dim=1)  # [B, L, n_vars], dtype=torch.complex
        freq = freq / x_enc.shape[1]

        # Frequency Normalization
        means = torch.mean(freq, dim=1)
        freq_abs = torch.abs(freq)
        stdev = torch.sqrt(torch.var(freq_abs, dim=1, keepdim=True))
        freq = (freq - means.unsqueeze(1).detach()) / stdev

        freq_pred = self.model(freq)

        # Frequency De-Normalization
        freq_pred = freq_pred * stdev
        freq_pred = freq_pred + means.unsqueeze(1).detach()


        freq_pred = freq_pred * freq_pred.shape[1]
        # pred_seq = torch.fft.irfft(freq_pred, dim=1).real[:, -self.configs.pred_len:]
        pred_seq = torch.fft.irfft(freq_pred, dim=1)[:, -self.configs.pred_len:]
        return pred_seq

class Model(nn.Module): # 时域处理模型trend项，频域模型处理seasonal项
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs

        self.harmony_num = 3
        self.t_model = T_Block(configs)
        self.f_model = F_Block(configs)

        self.linear = nn.Linear(configs.seq_len // 2 + 1, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # return self.t_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        freq = torch.fft.rfft(x_enc - torch.mean(x_enc, dim=1, keepdim=True), dim=1)
        freq = torch.abs(freq)
        _freq = freq.clone()
        _freq[:, :3, :] = 0
        _freq[:, freq.shape[1] // self.harmony_num:, :] = 0

        max_amp, indices = torch.max(_freq, dim=1, keepdim=True)
        amp_sum = torch.zeros_like(max_amp).to(x_enc.device)

        for i in range(self.harmony_num):
            har = (i + 1) * indices
            har_value = torch.gather(freq, 1, har) ** 2
            amp_sum = amp_sum + har_value

        total_sum = torch.sum(freq ** 2, dim=1, keepdim=True) + 1e-5
        weights = amp_sum / total_sum



        t_out = self.t_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        f_out = self.f_model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        out = f_out * weights + t_out * (1 - weights)


        return out
