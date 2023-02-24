# -- coding: utf-8 --
# @Time : 2023/2/10
# @Author : Ma Yunyi
# @Software: PyCharm

import torch.nn as nn
from dgl.utils import expand_as_pair
import dgl.function as fn
import torch


class MessageFunc(nn.Module):
    '''
        次序聚合中的消息传递函数的构造
    '''
    def __init__(self, orderInfo):
        super(MessageFunc, self).__init__()
        self.orderInfo = orderInfo

    def getMessageFun(self, feat_src, orderInfo):
        unbind_feat_src = torch.unbind(feat_src)
        unbind_orderInfo = torch.unbind(orderInfo)
        messageList = list(
            map(lambda x: torch.index_select(input=x[0], dim=0, index=x[1]),
                tuple(zip(unbind_feat_src, unbind_orderInfo))))
        mailboxInfo = torch.stack(messageList).view(-1, feat_src.shape[2])
        return mailboxInfo

    def forward(self, edges):
        feat_src = edges.src['embedding']  # 根据有链接的边，获得path表示
        mask_node_feat = self.getMessageFun(feat_src=feat_src,
                                            orderInfo=self.orderInfo)
        return {'m': mask_node_feat * edges.data['_edge_weight']}


class InterPathModel(nn.Module):
    '''
        根据次序聚合节点信息
    '''
    def __init__(self):
        super(InterPathModel, self).__init__()
        self._allow_zero_in_degree = True

    def forward(self, graph, feat, orderInfo):
        with graph.local_scope():
            graph.apply_edges(fn.copy_src(src='predFlow',out='_edge_weight'))
            aggregate_fn = MessageFunc(orderInfo=orderInfo)
            feat_src, feat_dst = expand_as_pair(feat, graph)

            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            return rst.squeeze(dim=1)