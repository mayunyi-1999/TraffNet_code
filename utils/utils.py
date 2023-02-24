import torch
import numpy as np

def expendDyInitFeature(x, max_size):
    padding = torch.zeros(((max_size - x.shape[0]), x.shape[1]))
    x_new = torch.vstack((x, padding))
    return x_new

def expendDyInitFeaturePadInf(x, max_size):
    padding = torch.ones(((max_size - x.shape[0]), x.shape[1])) * torch.tensor(float('inf'))
    x_new = torch.vstack((x, padding))
    return x_new

def getPathFeat(numpathnode, PathNodeDict_reverse, max_size, segment_feats, segmentNodeDict):
    path_feat_Tensor = torch.zeros(1, max_size, segment_feats.shape[1])
    for i in range(numpathnode):
        path_t = PathNodeDict_reverse[i]
        first_line = torch.zeros(1, segment_feats.shape[1])
        for segment in path_t:
            first_line = torch.vstack((first_line, segment_feats[segmentNodeDict[segment]]))
        path_feature = first_line[1:]

        expendFeature = expendDyInitFeature(path_feature, max_size)
        path_feat_Tensor = torch.vstack((path_feat_Tensor, expendFeature.unsqueeze(dim=0)))
    return path_feat_Tensor[1:]


def getPathFeatPadWithInf(numpathnode, PathNodeDict_reverse, max_size, segment_feats, segmentNodeDict):
    path_feat_Tensor = torch.zeros(1, max_size, segment_feats.shape[1])
    for i in range(numpathnode):
        path_t = PathNodeDict_reverse[i]
        first_line = torch.zeros(1, segment_feats.shape[1])
        for segment in path_t:
            first_line = torch.vstack((first_line, segment_feats[segmentNodeDict[segment]]))
        path_feature = first_line[1:]

        expendFeature = expendDyInitFeaturePadInf(path_feature, max_size)
        path_feat_Tensor = torch.vstack((path_feat_Tensor, expendFeature.unsqueeze(dim=0)))
    return path_feat_Tensor[1:]

def getPathFeat_taxibj(numpathnode, PathNodeDict_reverse, max_size, segment_feats):
    path_feat_Tensor = torch.zeros(1, max_size, segment_feats.shape[1])
    for i in range(numpathnode):
        path_t = PathNodeDict_reverse[i]
        first_line = torch.zeros(1, segment_feats.shape[1])
        for segment in path_t:
            first_line = torch.vstack((first_line, segment_feats[segment]))
        path_feature = first_line[1:]

        expendFeature = expendDyInitFeature(path_feature, max_size)
        path_feat_Tensor = torch.vstack((path_feat_Tensor, expendFeature.unsqueeze(dim=0)))
    return path_feat_Tensor[1:]


def getPathFeatPadWithInf_taxibj(numpathnode, PathNodeDict_reverse, max_size, segment_feats):
    path_feat_Tensor = torch.zeros(1, max_size, segment_feats.shape[1])
    for i in range(numpathnode):
        path_t = PathNodeDict_reverse[i]
        first_line = torch.zeros(1, segment_feats.shape[1])
        for segment in path_t:
            first_line = torch.vstack((first_line, segment_feats[segment]))
        path_feature = first_line[1:]

        expendFeature = expendDyInitFeaturePadInf(path_feature, max_size)
        path_feat_Tensor = torch.vstack((path_feat_Tensor, expendFeature.unsqueeze(dim=0)))
    return path_feat_Tensor[1:]






def getSeqMaxSize(pathNodeDictFileName):
    with open(pathNodeDictFileName, 'r') as f:
        pathNodeDict = eval(f.readlines()[0])

    allPath = list(pathNodeDict.keys())
    seqMaxSize = max([len(path) for path in allPath])

    return seqMaxSize

def getOd2Path_t(t, total_path_count_t, allOd_dict, allPath_dict):
    od2path = []
    path_t = list(total_path_count_t[t].keys())
    for path in path_t:
        od = (path[0], path[-1])
        od2path.append([allOd_dict[od], allPath_dict[path]])
    od2path_array = np.array(od2path)
    return od2path_array

