import torch
import torch.nn as nn
import torch.nn.functional as F

class TemperalGRU_sumovs(nn.Module):
    def __init__(self,
                 in_size,
                 hidden_size,
                 num_layers,
                 window_width,
                 batch_size,
                 edge_num,
                 horizon):
        super(TemperalGRU_sumovs, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        self.window_width = window_width
        self.batch_size = batch_size
        self.edge_num = edge_num

        self.gru = nn.GRU(input_size=in_size * self.edge_num,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True).to(torch.device('cuda:2'))

        self.pred = nn.Sequential(
            nn.Linear(hidden_size, 5500),
            nn.ReLU(),
            nn.Linear(5500, 5000),
            nn.ReLU(),
            nn.Linear(5000, 4500),
            nn.ReLU(),
            nn.Linear(4500, 4000),
            nn.ReLU(),
            nn.Linear(4000, 3500),
            nn.ReLU(),
            nn.Linear(3500, 3000),
            nn.ReLU(),
            nn.Linear(3000, 2500),
            nn.ReLU(),
            nn.Linear(2500, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1500),
            nn.ReLU(),
            nn.Linear(1500, 900),
            nn.ReLU(),
            nn.Linear(900, 600),
            nn.ReLU(),
            nn.Linear(600, self.edge_num),
            nn.ReLU(),
        ).to(torch.device('cuda:0'))

        self.linear_trans = nn.Linear(in_features=self.hidden_size,
                                      out_features=self.edge_num,
                                      bias=False).to(torch.device('cuda:0'))

    def forward(self, seq):
        seq = seq.reshape(-1, self.window_width, self.in_size * self.edge_num)
        all_out, _ = self.gru(seq)
        out = all_out[:, -1, :].to(torch.device('cuda:0'))  # 因为batch_first,要的是最后一个seq
        pred_out = self.pred(out)
        wx = self.linear_trans(out)
        pred_out = F.relu(pred_out + wx)
        return pred_out
