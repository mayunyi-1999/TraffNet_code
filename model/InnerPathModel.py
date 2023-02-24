# -- coding: utf-8 --
# @Time : 2023/2/10
# @Author : Ma Yunyi
# @Software: PyCharm

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

class InnerPathModel(nn.Module):
    def __init__(self,
                 seq_dim,
                 hidden_size,
                 num_layers,
                 seq_max_len,
                 edge_num):
        super(InnerPathModel, self).__init__()
        self.in_size = seq_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_max_len = seq_max_len
        self.edge_num = edge_num

        self.gru = nn.GRU(input_size=seq_dim,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bias=True,
                          bidirectional=True)

    def forward(self, seq, lengths):
        package = pack_padded_sequence(seq,lengths.cpu(),batch_first=True,enforce_sorted=False)
        results, _ = self.gru(package)
        outputs, lens = pad_packed_sequence(results,
                                            batch_first=True,total_length=seq.shape[1])
        return outputs