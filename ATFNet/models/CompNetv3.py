import torch
import torch.nn as nn
import torch.fft
from torch.nn.functional import relu
from layers.ComplexLayers import ComplexLinear, ComplexAttentionLayer, ComplexEncoderLayer, ComplexEncoder, complex_dropout


def complex_mse(pred, label):
    delta = pred - label
    return torch.mean(torch.abs(delta) ** 2)
def complex_relu(input):
    return relu(input.real).type(torch.complex64) + 1j * relu(input.imag).type(torch.complex64)




class CompEncoderBlock(nn.Module):
    # Input: DFT(input sequence), shape: [B, L:seq_len/2, n_vars] dtype: torch.cfloat
    # Output: DFT(input sequence + pred sequence), shape: [B, L:(seq_len+pred_len)/2, n_vars] dtype: torch.cfloat
    def __init__(self, configs):
        super(CompEncoderBlock, self).__init__()
        self.configs = configs
        self.device = configs.device
        self.is_emb = True

        self.ori_len = int((configs.seq_len + configs.pred_len) / 2) + 1
        self.tar_len = int((configs.seq_len + configs.pred_len) / 2) + 1
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

    def forward(self, x):  # x.shape: [B, L:seq_len/2, n_vars] dtype: torch.cfloat
        x = x.permute(0, 2, 1)
        if self.is_emb:
            x = self.emb(x)
        x = complex_dropout(x, self.dropout)
        x = self.encoder(x) # x.shape: [B, n_vars, d_model]
        x = self.projection(x)
        return x.permute(0, 2, 1)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.model = CompEncoderBlock(configs)

        self.low_pass = configs.low_pass
        self.threshold = configs.low_pass_threshold
        self.freq_threshold = int((configs.seq_len + configs.pred_len) / (2 * self.threshold))

        self.minus_mean = False




    def fft_cut(self, freq, N):
        if N % 2 == 0:
            ans = int(N / 2)
        else:
            ans = int(N / 2) - 1

        return freq[:, :ans + 1]

    def fft_reconstruct(self, freq, N):  # freq.shape: [B, N/2, n_vars]

        prolong = torch.flip(freq, dims=[1])[:, :-1, :].to(freq.device)
        if N % 2 == 0:
            prolong = prolong[:, 1:, :]
        prolong = torch.conj(prolong)

        ret = torch.cat((freq, prolong), dim=1)
        return ret

    def forward(self, x_enc, x_enc_mark, x_dec, x_dec_mark):
        if self.minus_mean:
            t_means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - t_means
        paddings = torch.zeros((x_enc.shape[0], self.pred_len ,x_enc.shape[2])).to(x_enc.device)
        x_enc = torch.concatenate((x_enc, paddings), dim=1)
        freq = torch.fft.rfft(x_enc, dim=1)  # [B, L, n_vars], dtype=torch.complex
        freq = freq / x_enc.shape[1]
        # freq = self.fft_cut(freq, x_enc.shape[1])

        # Frequency Normalization
        means = torch.mean(freq, dim=1)
        freq_abs = torch.abs(freq)
        stdev = torch.sqrt(torch.var(freq_abs, dim=1, keepdim=True))
        freq = (freq - means.unsqueeze(1).detach()) / stdev

        freq_pred = self.model(freq)

        # Frequency De-Normalization
        freq_pred = freq_pred * stdev
        freq_pred = freq_pred + means.unsqueeze(1).detach()


        if self.configs.low_pass:
            filter = torch.ones_like(freq_pred)
            filter[:, self.freq_threshold:] = 0 + 0j
            freq_pred = freq_pred * filter

        # freq_pred = self.fft_reconstruct(freq_pred, self.configs.pred_len + self.configs.seq_len)
        freq_pred = freq_pred * freq_pred.shape[1]
        # pred_seq = torch.fft.irfft(freq_pred, dim=1).real[:, -self.configs.pred_len:]
        pred_seq = torch.fft.irfft(freq_pred, dim=1)[:, -self.configs.pred_len:]
        if self.configs.tf_loss:
            pred_seq = torch.fft.ifft(freq_pred, dim=1).real
        if self.configs.freq_loss:
            return freq_pred
        if self.minus_mean:
            pred_seq = pred_seq + t_means.detach()
        return pred_seq


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=216, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--complex_dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--low_pass', type=bool, default=False)  # 是否进行低通滤波
    parser.add_argument('--low_pass_threshold', type=float, default=3)  # 低通滤波周期最低值
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--freq_loss', type=bool, default=True)  # 是否在频域计算loss，若为False则在ifft回时域后计算loss
    parser.add_argument('--fnet_d_ff', type=int, default=2048, help='dimension of fcn of fnet')
    parser.add_argument('--fnet_d_model', type=int, default=512, help='dimension of model of fnet')
    parser.add_argument('--exclude_dc', type=bool, default=False)  # 是否去除直流分量再过模型
    parser.add_argument('--model', type=str, default='CompNet')

    args = parser.parse_args(args=[])
    model = Model(args)
    input_seq = torch.randn((32, 216, 7))
    out = model(input_seq)
    label_r = torch.randn((32, 156))
    label_i = torch.randn((32, 156))
    label = torch.complex(label_r, label_i)
    loss_fn = complex_mse
    loss = loss_fn(out, label)
    loss.backward()
    print(out.shape)
