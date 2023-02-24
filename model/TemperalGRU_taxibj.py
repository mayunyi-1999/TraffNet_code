import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperalGRUModelTaxiBJ(nn.Module):
    def __init__(self,
                 in_size,
                 hidden_size,
                 num_layers,
                 window_width,
                 batch_size,
                 edge_num,
                 horizon,
                 device):
        super(TemperalGRUModelTaxiBJ, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.window_width = window_width
        self.batch_size = batch_size
        self.device = device
        self.edge_num = edge_num
        self.horizon = horizon

        self.gru = nn.GRU(input_size=in_size * self.edge_num,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)

        self.pred = nn.Sequential(
            nn.Linear(hidden_size, 2400),
            nn.ReLU(),
            nn.Linear(2400, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1600),
            nn.ReLU(),
            nn.Linear(1600, 1200),
            nn.ReLU(),
            nn.Linear(1200, 800),
            nn.ReLU(),
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Linear(400, self.edge_num),
        )

        self.linear_trans = nn.Linear(in_features=self.hidden_size,
                                      out_features=self.edge_num,
                                      bias=False)

    def forward(self, seq):
        seq = seq.reshape(-1, self.window_width, self.in_size * self.edge_num)
        pad = torch.zeros(seq.shape[0], self.horizon - 1, seq.shape[-1]).to(self.device)
        pad_seq = torch.hstack((seq, pad))
        all_out, _ = self.gru(pad_seq)
        out = all_out[:, all_out.shape[1] - self.horizon:all_out.shape[1], :]  # 因为batch_first,要的是最后一个seq
        pred_out = self.pred(out)
        pred_out = pred_out.view(-1, self.horizon, self.edge_num)
        wx = self.linear_trans(out)
        res_out = F.relu(pred_out + wx)
        return res_out