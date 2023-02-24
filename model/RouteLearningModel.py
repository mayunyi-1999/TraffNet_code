# -- coding: utf-8 --
# @Time : 2023/2/10
# @Author : Ma Yunyi
# @Software: PyCharm

import torch.nn as nn
import dgl.nn.pytorch as dglnn
import dgl
import torch
import dgl.function as fn
from dgl.nn.functional import edge_softmax

class od2PathNumModel(nn.Module):
    '''
        路径选择模块中的消息传递机制
    '''
    def __init__(self):
        super(od2PathNumModel, self).__init__()

    def forward(self,graph,get_selectPro=False):
        with graph.local_scope():
            aggfn = fn.copy_src(src='gatEmb', out='he')
            graph.apply_edges(aggfn,etype='select-')

            edata_dict = {('od', 'select+', 'path'): graph.edata['he'][('path', 'select-', 'od')],
                          ('path', 'select-', 'od'): graph.edata['he'][('path', 'select-', 'od')]}
            selectProb = edge_softmax(graph,edata_dict)[('path','select-','od')]

            graph.edges['select+'].data['selectProb'] = selectProb


            graph.multi_update_all({'select+':(fn.u_mul_e('odNum','selectProb','m'),
                                               fn.sum('m','od2PathNum'))},'sum')

            rst = graph.nodes['path'].data['od2PathNum']
            if get_selectPro == True:
                return rst,selectProb
            else:
                return rst


class RouteLearningModel(nn.Module):
    '''
        路径选择模块
    '''
    def __init__(self, in_feats, out_feats, num_heads,seq_max_len):
        super(RouteLearningModel, self).__init__()
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.seq_max_len = seq_max_len

        self.gatconv = dglnn.GATConv(in_feats=in_feats,
                                     out_feats=out_feats,
                                     num_heads=num_heads,
                                     allow_zero_in_degree=True)

        self.linear = nn.Sequential(nn.Linear(out_feats * num_heads* seq_max_len, 180),
                                    nn.ReLU(),
                                    nn.Linear(180, 150),
                                    nn.ReLU(),
                                    nn.Linear(150, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 80),
                                    nn.ReLU(),
                                    nn.Linear(80, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, 1))


        self.linear_trans = nn.Linear(in_features=out_feats * num_heads*seq_max_len,
                                      out_features=1,
                                      bias=False)

        self.od2PathNumModel = od2PathNumModel()

    def forward(self, graph):
        g_sub = dgl.metapath_reachable_graph(graph, ['select-', 'select+'])
        routeLearningEmbed = g_sub.ndata['embedding'].to(torch.float32)

        # 1.获取路径选择表示
        g_sub = dgl.add_self_loop(g_sub)
        gatEmb = self.gatconv(g_sub, routeLearningEmbed)
        gatEmb = gatEmb.view(-1, self.out_feats * self.num_heads *self.seq_max_len)
        gatEmb_nor = self.linear(gatEmb)
        wx = self.linear_trans(gatEmb)
        graph.nodes['path'].data['gatEmb'] = gatEmb_nor + wx
        # 2.获得根据od数量和路径选择推算path的数量
        # 获得od和path之间的双向二部图 --->获取od和path之间的路径选择
        g_select = graph.edge_type_subgraph(['select-', 'select+'])
        predFlow, selectProb = self.od2PathNumModel(g_select, get_selectPro=True)
        return predFlow, selectProb
