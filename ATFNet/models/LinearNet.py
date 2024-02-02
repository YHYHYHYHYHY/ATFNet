import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff

        self.Linear1 = nn.Linear(self.seq_len, self.d_model)
        self.Linear2 = nn.Linear(self.d_model, self.seq_len)
        self.Linear3 = nn.Linear(self.seq_len, self.pred_len)

        self.relu = nn.ReLU()

    def forward(self, x, x_mark, dec, dec_mark):
        x = x.view(x.shape[0], -1)
        # Norm
        means = torch.mean(x, dim=1).detach().unsqueeze(-1)
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True))
        x = (x - means) / std

        _x = x.clone()
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.Linear2(x)
        x = self.relu(x)
        x = x + _x
        x = self.Linear3(x)

        # De-Norm
        x = x * std + means
        return x.unsqueeze(-1)


