import dgl
import torch
from torch.utils.data import Dataset

def collect_fn_preSelPath(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.vstack(labels)

def getDataGenerate(t):
    graph = dgl.load_graphs(f'../HTGWithFeat_sumosy/Hetero_{t}.bin', [0])[0][0]
    label = graph.nodes['path'].data['pathNum']
    print(f'{t} is ok')
    return (graph,label)


class PreTrainRouteLearningDataset(Dataset):
    def __init__(self,
                 datasetType,
                 timestamps,
                 window_width, out_len,
                 start_idx):
        if datasetType == 'Train' or datasetType == 'Val':
            self.data = [getDataGenerate(t) for t in range(start_idx,start_idx+timestamps)]
        elif datasetType == 'Test':
            self.data = [getDataGenerate(t) for t in range(start_idx,start_idx+timestamps-window_width-out_len)]
        else:
            raise ValueError("Invalid datasetType: {}".format(datasetType))

        self.len = len(self.data)

    def __getitem__(self, index):
        return (self.data[index][0], self.data[index][1])

    def __len__(self):
        return self.len

