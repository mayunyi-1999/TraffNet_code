# -- coding: utf-8 --
# @Time : 2023/2/10
# @Author : Ma Yunyi
# @Software: PyCharm
import numpy as np
import torch
from torch.utils.data import Dataset
import dgl
import os
import pathlib


def getDataGenerateOneStep(idx, labels, horizon):
    label = labels[idx:idx + horizon].squeeze(axis=1)
    folder = pathlib.Path(__file__).parent.parent.resolve()
    graphPath = os.path.join(folder, 'batchHtgGraph_taxibj')
    batchGraph = dgl.load_graphs(f'{graphPath}/batchHtg{idx}.bin', [0])[0][0]
    print(f'{idx} is ok')
    return (batchGraph, label)


def traffNet_collect_fn(samples):
    graphSeqTimeWindows, labels = map(list, zip(*samples))
    graphSeq = []
    for graph in graphSeqTimeWindows:
        graphSeq.append(graph)
    batched_graph = dgl.batch(graphSeq)
    return batched_graph, torch.tensor(labels)


class DataSetTaxiBJ(Dataset):
    def __init__(self, datasetType,
                 timestamps,
                 window_width,
                 horizon,
                 start_idx):
        self.labels = np.load( '../data_taxibj/labelsNp_taxibj.npz')['arr_0']

        if datasetType == 'Train' or datasetType == 'Val':
            self.data = [getDataGenerateOneStep(idx=t,
                                                labels=self.labels,
                                                horizon=horizon) for t in range(start_idx, start_idx + timestamps)]

        elif datasetType == 'Test':
            self.data = [getDataGenerateOneStep(idx=t,
                                                labels=self.labels,
                                                horizon=horizon) for t in
                         range(start_idx, start_idx + timestamps - window_width - horizon)]
        else:
            raise ValueError("Invalid datasetType: {}".format(datasetType))

    def __getitem__(self, index):
        return (self.data[index][0], self.data[index][1])

    def __len__(self):
        return len(self.data)
