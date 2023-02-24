# -- coding: utf-8 --
# @Time : 2023/2/10
# @Author : Ma Yunyi
# @Software: PyCharm

import torch.nn as nn
import torch
import dgl

from model.InnerPathModel import InnerPathModel
from model.InterPathModel import InterPathModel
from model.RouteLearningModel import RouteLearningModel
from model.TemperalGRU_sumosy import TemperalGRU_sumosy



class PredTraffModel_sumosy(nn.Module):
    def __init__(self, seq_max_len, edge_num, window_width,batch_size, horizon):
        super(PredTraffModel_sumosy, self).__init__()
        self.seq_max_len = seq_max_len
        self.edge_num = edge_num
        self.window_width = window_width
        self.horizon = horizon
        self.batch_size = batch_size

        self.embedding = nn.Embedding(num_embeddings=edge_num + 1,
                                      embedding_dim=48).to(torch.device('cuda:0'))

        self.InnerPathModel = InnerPathModel(seq_dim=48 + 5,
                                             hidden_size=280,
                                             num_layers=2,
                                             seq_max_len=self.seq_max_len,
                                             edge_num=self.edge_num).to(torch.device('cuda:0'))

        self.RouteLearningModel = RouteLearningModel(in_feats=280 * 2,
                                                     out_feats=1024,
                                                     num_heads=2,
                                                     seq_max_len=self.seq_max_len).to(torch.device('cuda:0'))

        self.interPathModel = InterPathModel().to(torch.device('cuda:0'))

        self.gru = TemperalGRU_sumosy(in_size=280 * 2,
                               hidden_size=1200,
                               num_layers=2,
                               window_width=self.window_width,
                               batch_size=self.batch_size,
                               edge_num=self.edge_num,
                               horizon=self.horizon)

    def forward(self, batch_graph):
        edgeIdOfPath = batch_graph.nodes['path'].data['edgeId']
        edgeIdOfPath = edgeIdOfPath.squeeze(dim=-1).to(torch.int64)
        lengths = torch.count_nonzero(edgeIdOfPath, dim=1).float()
        pathFlow = batch_graph.nodes['path'].data['pathNum']

        pathEdgeEmbedding = self.embedding(edgeIdOfPath).to(torch.device('cuda:0'))

        # 在inner-path双向gru推演时候考虑path由edge属性构成特征的影响
        # 1. get inner-path embedding
        pathedgeFeat = batch_graph.nodes['path'].data['pathSegmentFeat']
        pathInputEmbedding = torch.cat([pathEdgeEmbedding, pathedgeFeat], dim=2)
        pathEmbedding = self.InnerPathModel(pathInputEmbedding, lengths).to(torch.device('cuda:0'))  # emb dim + feature dim

        # 2. route learning
        batch_graph.nodes['path'].data['embedding'] = pathEmbedding
        predFlow, _ = self.RouteLearningModel(batch_graph)
        batch_graph.nodes['path'].data['predFlow'] = predFlow

        # 3. inter-path embedding
        G_predict = dgl.metapath_reachable_graph(batch_graph, ['pass+'])
        orderInfo = batch_graph.edges['pass+'].data['orderInfo']
        edge_feats = batch_graph.nodes['edge'].data['feature'].float()

        path2edge = self.interPathModel(graph=G_predict,
                                        feat=(pathEmbedding, edge_feats),
                                        orderInfo=orderInfo).to(torch.device('cuda:1'))

        # 4. temperal GRU
        out = self.gru(path2edge)
        return out, predFlow, pathFlow
