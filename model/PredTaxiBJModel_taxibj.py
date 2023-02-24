import torch.nn as nn
import torch
import dgl

from model.InnerPathModel import InnerPathModel
from model.InterPathModel import InterPathModel
from model.RouteLearningModel import RouteLearningModel
from model.TemperalGRU_taxibj import TemperalGRUModelTaxiBJ


class PredTraffModel_taxibj(nn.Module):
    def __init__(self, seq_max_len, edge_num, window_width, batch_size, horizon,device):
        super(PredTraffModel_taxibj, self).__init__()
        self.seq_max_len = seq_max_len
        self.edge_num = edge_num
        self.window_width = window_width
        self.horizon = horizon
        self.batch_size = batch_size
        self.device = device

        self.embedding = nn.Embedding(num_embeddings=edge_num + 1,
                                      embedding_dim=48)

        self.InnerPathModel = InnerPathModel(seq_dim=48 + 2,
                                             hidden_size=320,
                                             num_layers=2,
                                             seq_max_len=self.seq_max_len,
                                             edge_num=self.edge_num)

        self.RouteLearningModel = RouteLearningModel(in_feats=320 * 2,
                                                     out_feats=1024,
                                                     num_heads=2,
                                                     seq_max_len=self.seq_max_len)

        self.interPathModel = InterPathModel()

        self.gru = TemperalGRUModelTaxiBJ(in_size=320 * 2,
                                          hidden_size=1200,
                                          num_layers=2,
                                          window_width=self.window_width,
                                          batch_size=self.batch_size,
                                          edge_num=self.edge_num,
                                          horizon=self.horizon,
                                          device=self.device)

    def forward(self, batch_graph):
        edgeIdOfPath = batch_graph.nodes['path'].data['segmentId']
        edgeIdOfPath = edgeIdOfPath.squeeze(dim=-1).to(torch.int64)
        lengths = torch.count_nonzero(edgeIdOfPath, dim=1).float()
        pathFlow = batch_graph.nodes['path'].data['pathNum']

        pathEdgeEmbedding = self.embedding(edgeIdOfPath)

        # 在inner-path双向gru推演时候考虑path由edge属性构成特征的影响
        # 1. get inner-path embedding
        pathedgeFeat = batch_graph.nodes['path'].data['pathSegmentFeat']
        pathInputEmbedding = torch.cat([pathEdgeEmbedding, pathedgeFeat], dim=2)
        pathEmbedding = self.InnerPathModel(pathInputEmbedding, lengths)  # emb dim + feature dim
        # batch_graph.nodes['path'].data['embedding'] = pathEmbedding

        # 2. route learning
        batch_graph.nodes['path'].data['embedding'] = pathEmbedding
        predFlow, _ = self.RouteLearningModel(batch_graph)
        batch_graph.nodes['path'].data['predFlow'] = predFlow

        # 3. inter-path embedding
        G_predict = dgl.metapath_reachable_graph(batch_graph, ['pass+'])
        orderInfo = batch_graph.edges['pass+'].data['orderInfo']
        edge_feats = batch_graph.nodes['segment'].data['feature'].float()
        # staticInfo = edge_feats
        path2edge = self.interPathModel(graph=G_predict,
                                        feat=(pathEmbedding, edge_feats),
                                        orderInfo=orderInfo)

        # 4. temperal GRU
        out = self.gru(path2edge)
        return out, predFlow, pathFlow
