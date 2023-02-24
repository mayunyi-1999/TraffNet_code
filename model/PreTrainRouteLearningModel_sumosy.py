import torch.nn as nn
import torch

from model.InnerPathModel import InnerPathModel
from model.RouteLearningModel import RouteLearningModel


class PretrainRouteLearningModel(nn.Module):
    def __init__(self, seq_max_len, edge_num):
        super(PretrainRouteLearningModel, self).__init__()
        self.seq_max_len = seq_max_len
        self.edge_num = edge_num

        self.embedding = nn.Embedding(num_embeddings=edge_num + 1,
                                      embedding_dim=48)

        self.InnerPathModel = InnerPathModel(seq_dim=48 + 5,
                                             hidden_size=280,
                                             num_layers=2,
                                             seq_max_len=self.seq_max_len,
                                             edge_num=self.edge_num)

        self.RouteLearningModel = RouteLearningModel(in_feats=280 * 2,
                                                     out_feats=1024,
                                                     num_heads=2,
                                                     seq_max_len=self.seq_max_len)

    def forward(self, batch_graph):
        edgeIdOfPath = batch_graph.nodes['path'].data['segmentId']
        edgeIdOfPath = edgeIdOfPath.squeeze(dim=-1).to(torch.int64)
        lengths = torch.count_nonzero(edgeIdOfPath, dim=1).float()
        pathEdgeEmbedding = self.embedding(edgeIdOfPath)

        # 在innerpath双向gru推演时候考虑path由edge属性构成特征的影响
        pathedgeFeat = batch_graph.nodes['path'].data['pathSegmentFeat']
        pathInputEmbedding = torch.cat([pathEdgeEmbedding, pathedgeFeat], dim=2)
        pathEmbedding = self.InnerPathModel(pathInputEmbedding, lengths)

        batch_graph.nodes['path'].data['embedding'] = pathEmbedding

        predFlow, _ = self.RouteLearningModel(batch_graph)
        return predFlow, _
