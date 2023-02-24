import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperalGRU_sumosy(nn.Module):
    def __init__(self,
                 in_size,
                 hidden_size,
                 num_layers,
                 window_width,
                 batch_size,
                 edge_num,
                 horizon,
                 ):
        super(TemperalGRU_sumosy, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.window_width = window_width
        self.batch_size = batch_size
        self.edge_num = edge_num
        self.horizon = horizon

        self.gru = nn.GRU(input_size=in_size * self.edge_num,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True).to(torch.device('cuda:2'))

        self.pred = nn.Sequential(
                nn.Linear(hidden_size, 4600),
                nn.ReLU(),
                nn.Linear(4600, 4000),
                nn.ReLU(),
                nn.Linear(4000, 3600),
                nn.ReLU(),
                nn.Linear(3600, 3000),
                nn.ReLU(),
                nn.Linear(3000, 2400),
                nn.ReLU(),
                nn.Linear(2400, 1800),
                nn.ReLU(),
                nn.Linear(1800, 1300),
                nn.ReLU(),
                nn.Linear(1300, 1000),
                nn.ReLU(),
                nn.Linear(1000, 650),
                nn.ReLU(),
                nn.Linear(650, self.edge_num),
            ).to(torch.device('cuda:0'))

        self.linear_trans = nn.Linear(in_features=self.hidden_size,
                                      out_features=self.edge_num,
                                      bias=False).to(torch.device('cuda:0'))

    def forward(self, seq):
        seq = seq.reshape(-1, self.window_width, self.in_size * self.edge_num)
        pad = torch.zeros(seq.shape[0], self.horizon - 1, seq.shape[-1]).to(torch.device('cuda:1'))
        pad_seq = torch.hstack((seq, pad))
        all_out, _ = self.gru(pad_seq)
        out = all_out[:, all_out.shape[1] - self.horizon:all_out.shape[1], :].to(torch.device('cuda:0')) # 因为batch_first,要的是最后一个seq
        pred_out = self.pred(out)
        pred_out = pred_out.view(-1, self.horizon, self.edge_num)
        wx = self.linear_trans(out)
        res_out = F.relu(pred_out + wx)
        return res_out