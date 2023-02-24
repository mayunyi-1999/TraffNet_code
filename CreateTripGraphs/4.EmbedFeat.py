# -- coding: utf-8 --
# @Time : 2023/2/10 15:06
# @Author : Ma Yunyi
# @Software: PyCharm

import torch
from dgl import save_graphs, load_graphs
import numpy as np
import pandas as pd
from sklearn import preprocessing

from utils.utils import getPathFeat, getPathFeatPadWithInf, getOd2Path_t, getSeqMaxSize, getPathFeatPadWithInf_taxibj, \
    getPathFeat_taxibj

'''
    -------------获取每个Path的特征----------------
'''


def EmbedFeat_sumovs(startTimestamp, endtimestamps, seqmaxsize, allTimestamps):
    # 1. 数据读取
    df_flow = pd.read_csv("../data_sumovs/realflowAll14dayswhatIf.csv")
    df_avgspeed = pd.read_csv("../data_sumovs/avgspeedAll14dayswhatIf.csv")
    df_limitSpeed = pd.read_csv('../data_sumovs/vslimitSpeed.csv')
    df_staticInfo = pd.read_csv("../data_sumovs/staticInfo.csv")[['segment_numLanes', 'segment_length']]
    df_path2areaOrder = pd.read_csv('../data_sumovs/path2segment_sumovs.csv')

    total_flow = np.array(df_flow).flatten('C').reshape(-1, 1)
    total_avgspeed = np.array(df_avgspeed).flatten('C').reshape(-1, 1)
    total_limitspeed = np.array(df_limitSpeed).flatten('C').reshape(-1, 1)
    total_staticInfo = np.tile(np.array(df_staticInfo), (allTimestamps, 1))

    totalsegmentFeat = np.concatenate((total_flow, total_avgspeed, total_limitspeed, total_staticInfo), axis=1)
    max_min_scaler = preprocessing.MinMaxScaler()
    minmax_feat = max_min_scaler.fit_transform(totalsegmentFeat)

    with open("../data_sumovs/total_path_count_tAll14dayswhatIf.txt", 'r') as f:
        total_path_count_t = eval(f.readlines()[0])

    with open("../data_sumovs/segmentNodeDict_sumovs.txt", "r") as f:
        segmentNodeDict = eval(f.readlines()[0])

    with open("../data_sumovs/total_od_count_tAll14dayswhatIf.txt", 'r') as f:
        total_od_count_t = eval(f.readlines()[0])

    with open("../data_sumovs/odNodeDict_sumovs.txt", "r") as f:
        allOd_dict = eval(f.readlines()[0])
        allOd_dict_reversed = {v: k for k, v in allOd_dict.items()}

    with open("../data_sumovs/pathNodeDict_sumovs.txt", "r") as f:
        pathNodeDict = eval(f.readlines()[0])
        PathNodeDict_reverse = {v: k for k, v in pathNodeDict.items()}
        pathLen = {index: len(path) for path, index in pathNodeDict.items()}

    with open('../data_sumovs/jam_segment_info_dict.txt', 'r') as f:
        jam_segment_info_dict = eval(f.readlines()[0])

    for t in range(startTimestamp, endtimestamps):
        g_t = load_graphs(f"../HtgNoFeat_sumovs/Hetero_{t}.bin", [0])[0][0]
        segment_num = g_t.num_nodes('segment')
        # 1. segment节点嵌入
        # 1.1 segment节点：未归一化的原始特征
        flow = torch.tensor(np.array(df_flow.iloc[t]).reshape(segment_num, 1).astype(float))
        limitspeed = torch.tensor(np.array(df_limitSpeed.iloc[t]).reshape(segment_num, 1).astype(float))
        avgspeed = torch.tensor(np.array(df_avgspeed.iloc[t]).reshape(segment_num, 1).astype(float))
        staticInfo = torch.tensor(np.array(df_staticInfo).reshape(segment_num, 2))
        segmentInitFeat = torch.cat([flow, avgspeed, limitspeed, staticInfo], dim=1)

        # 1.2 segment节点：归一化的特征
        g_t.nodes['segment'].data['feature'] = torch.tensor(minmax_feat[t * segment_num:(t + 1) * segment_num]).to(
            torch.float32)
        # 1.3 segment节点：id
        g_t.nodes['segment'].data['id'] = torch.arange(1, segment_num + 1).reshape(-1, 1)

        # 2.path特征嵌入
        segmentId = g_t.nodes['segment'].data['id']
        num_node_path = g_t.num_nodes("path")

        # 2.1 推演用到的id信息
        segmentIdOfPath = getPathFeat(numpathnode=num_node_path,
                                      PathNodeDict_reverse=PathNodeDict_reverse,
                                      max_size=seqmaxsize,
                                      segment_feats=segmentId,
                                      segmentNodeDict=segmentNodeDict)
        g_t.nodes['path'].data['segmentId'] = segmentIdOfPath

        # 2.2 路径选择path的信息
        pathInitFeats = getPathFeat(numpathnode=num_node_path,
                                    PathNodeDict_reverse=PathNodeDict_reverse,
                                    max_size=seqmaxsize,
                                    segment_feats=segmentInitFeat,
                                    segmentNodeDict=segmentNodeDict)

        pathInitFeatsInf = getPathFeatPadWithInf(numpathnode=num_node_path,
                                                 PathNodeDict_reverse=PathNodeDict_reverse,
                                                 max_size=seqmaxsize,
                                                 segment_feats=segmentInitFeat,
                                                 segmentNodeDict=segmentNodeDict)

        segmentMinMaxScalerFeat = g_t.nodes['segment'].data['feature']
        pathFeat = getPathFeat(numpathnode=num_node_path,
                               PathNodeDict_reverse=PathNodeDict_reverse,
                               max_size=seqmaxsize,
                               segment_feats=segmentMinMaxScalerFeat,
                               segmentNodeDict=segmentNodeDict)
        g_t.nodes['path'].data['pathSegmentFeat'] = pathFeat

        # [1.平均速度3个flim1,f2 f3,2.限速3个，车道数3个，长度 == 10个]
        # 2.2.1  平均车道数，平均限速
        pathAvgSpeedLaneLimSpeed = pathInitFeats.index_select(dim=2, index=torch.LongTensor([1, 2, 3]))
        pathAvgSpeedLaneLimSpeedSum = torch.sum(pathAvgSpeedLaneLimSpeed, dim=1)
        pathLenTensor = torch.tensor(list(pathLen.values())).reshape(num_node_path, 1)
        averagepathLaneLimSpeed = pathAvgSpeedLaneLimSpeedSum / pathLenTensor
        # 2.2.2 获取最大车道数
        pathAvgSpeedLaneLimSpeedLaneNum = pathInitFeats.index_select(dim=2, index=torch.LongTensor([1, 2, 3]))
        pathSpeedLaneLaneNumLimSpeedSumMax = torch.max(pathAvgSpeedLaneLimSpeedLaneNum, dim=1)[0]
        # 2.2.3 获取总长度
        pathLength = pathInitFeats.index_select(dim=2, index=torch.LongTensor([4]))
        pathSumLength = torch.sum(pathLength, dim=1)
        # 2.2.4 获取最小车道数
        pathLaneNumPadInf = pathInitFeatsInf.index_select(dim=2, index=torch.LongTensor([1, 2, 3]))
        pathMinavgSpeedLaneNumLimSpeedMin = torch.min(pathLaneNumPadInf, dim=1)[0]

        onePathFeature = torch.cat(
            [averagepathLaneLimSpeed, pathSpeedLaneLaneNumLimSpeedSumMax, pathMinavgSpeedLaneNumLimSpeedMin,
             pathSumLength],
            dim=1)
        onePathFeat = max_min_scaler.fit_transform(np.array(onePathFeature.float()))
        g_t.nodes['path'].data['onePathFeat'] = torch.tensor(onePathFeat)

        # 3. od特征嵌入
        num_node_od = g_t.num_nodes("od")
        od_count_t = total_od_count_t[t]
        if od_count_t != {}:
            od2path_t_array = getOd2Path_t(t, total_path_count_t, allOd_dict, pathNodeDict)
            od_id = od2path_t_array[0:, 0]
            odNum = torch.zeros(1, 1)
            for i in range(num_node_od):
                if i in od_id:
                    # od i 的数量：od_count_t[allOd_dict_reversed[i]]
                    odNum = torch.vstack((odNum, torch.tensor(od_count_t[allOd_dict_reversed[i]])))
                else:
                    odNum = torch.vstack((odNum, torch.tensor([0.])))
            odNum = odNum[1:]
            g_t.nodes['od'].data['odNum'] = odNum
        else:
            g_t.nodes['od'].data['odNum'] = torch.zeros((num_node_od, 1))

        # 4. 边上信息：path->segment嵌入order信息
        g_t.edges['pass+'].data['orderInfo'] = torch.tensor(df_path2areaOrder['orderInfo']).reshape(-1, 1)

        # 5. segment的whatif 标识
        jam_segment_list = []
        if t in jam_segment_info_dict:
            # what if
            for i in range(segment_num):
                if i in eval(jam_segment_info_dict[t]):
                    jam_segment_list.append(1)
                else:
                    jam_segment_list.append(0)
        else:
            # no whatif
            jam_segment_list = [0 for _ in range(segment_num)]

        g_t.nodes['segment'].data['isWhatif'] = torch.tensor(jam_segment_list).reshape(-1, 1)

        # 6. 给每个path嵌入path的数量
        path_count_t = total_path_count_t[t]
        if path_count_t != {}:
            od2path_t_array = getOd2Path_t(t, total_path_count_t, allOd_dict, pathNodeDict)
            path_id = od2path_t_array[0:, 1]
            pathNum = torch.zeros(1, 1)
            for i in range(num_node_path):
                if i in path_id:
                    # od i 的数量：od_count_t[allOd_dict_reversed[i]]
                    pathNum = torch.vstack((pathNum, torch.tensor(path_count_t[PathNodeDict_reverse[i]])))
                else:
                    pathNum = torch.vstack((pathNum, torch.tensor([0.])))
            pathNum = pathNum[1:]
            g_t.nodes['path'].data['pathNum'] = pathNum
        else:
            g_t.nodes['path'].data['pathNum'] = torch.zeros((num_node_path, 1))

        save_graphs(f"../HTGWithFeat_sumovs/Hetero_{t}.bin", [g_t])
        print(f'sumovs:{t} ,is over')


def EmbedFeat_sumosy(startTimestamp, endtimestamps, seqmaxsize, allTimestamps):
    # 1. 数据读取
    df_flow = pd.read_csv("../data_sumosy/realflowAll28days_sumosy.csv")
    df_staticInfo = pd.read_csv("../data_sumosy/staticInfo_sumosy.csv")
    df_staticInfo = df_staticInfo[['segment_numLanes', 'segment_speed', 'segment_length']]
    df_avgspeed = pd.read_csv("../data_sumosy/avgspeedAll28days_sumosy.csv")
    df_path2areaOrder = pd.read_csv('../data_sumosy/path2segment_sumosy.csv')

    total_flow = np.array(df_flow).flatten('C').reshape(-1, 1)
    total_avgspeed = np.array(df_avgspeed).flatten('C').reshape(-1, 1)
    total_staticInfo = np.tile(np.array(df_staticInfo), (allTimestamps, 1))
    totalsegmentFeat = np.concatenate((total_flow, total_avgspeed, total_staticInfo), axis=1)
    max_min_scaler = preprocessing.MinMaxScaler()
    minmax_feat = max_min_scaler.fit_transform(totalsegmentFeat)

    with open("../data_sumosy/total_path_count_tAll28days_sumosy.txt") as f:
        total_path_count_t = eval(f.readlines()[0])

    with open("../data_sumosy/segmentNodeDict_sumosy.txt", "r") as f:
        segmentNodeDict = eval(f.readlines()[0])

    with open("../data_sumosy/total_od_count_tAll28days_sumosy.txt", 'r') as f:
        total_od_count_t = eval(f.readlines()[0])

    with open("../data_sumosy/odNodeDict_sumosy.txt", "r") as f:
        allOd_dict = eval(f.readlines()[0])
        allOd_dict_reversed = {v: k for k, v in allOd_dict.items()}

    with open("../data_sumosy/pathNodeDict_sumosy.txt", "r") as f:
        pathNodeDict = eval(f.readlines()[0])
        PathNodeDict_reverse = {v: k for k, v in pathNodeDict.items()}
        pathLen = {index: len(path) for path, index in pathNodeDict.items()}

    for t in range(startTimestamp, endtimestamps):
        g_t = load_graphs(f"../HtgNoFeat_sumosy/Hetero_{t}.bin", [0])[0][0]
        segment_num = g_t.num_nodes('segment')
        # 1. segment节点嵌入
        # 1.1 segment节点：未归一化的原始特征
        flow = torch.tensor(np.array(df_flow.iloc[t]).reshape(segment_num, 1).astype(float))
        avgspeed = torch.tensor(np.array(df_avgspeed.iloc[t]).reshape(segment_num, 1).astype(float))
        staticInfo = torch.tensor(np.array(df_staticInfo).reshape(segment_num, 3))
        segmentInitFeat = torch.cat([flow, avgspeed, staticInfo], dim=1)

        # 1.2 segment节点：归一化的特征
        g_t.nodes['segment'].data['feature'] = torch.tensor(minmax_feat[t * segment_num:(t + 1) * segment_num]).to(
            torch.float32)
        # 1.3 segment节点：id
        g_t.nodes['segment'].data['id'] = torch.arange(1, segment_num + 1).reshape(-1, 1)

        # 2.path特征嵌入
        segmentId = g_t.nodes['segment'].data['id']
        num_node_path = g_t.num_nodes("path")

        # 2.1 推演用到的id信息
        segmentIdOfPath = getPathFeat(numpathnode=num_node_path,
                                      PathNodeDict_reverse=PathNodeDict_reverse,
                                      max_size=seqmaxsize,
                                      segment_feats=segmentId,
                                      segmentNodeDict=segmentNodeDict)
        g_t.nodes['path'].data['segmentId'] = segmentIdOfPath

        # 2.2 路径选择path的信息
        pathInitFeats = getPathFeat(numpathnode=num_node_path,
                                    PathNodeDict_reverse=PathNodeDict_reverse,
                                    max_size=seqmaxsize,
                                    segment_feats=segmentInitFeat,
                                    segmentNodeDict=segmentNodeDict)

        pathInitFeatsInf = getPathFeatPadWithInf(numpathnode=num_node_path,
                                                 PathNodeDict_reverse=PathNodeDict_reverse,
                                                 max_size=seqmaxsize,
                                                 segment_feats=segmentInitFeat,
                                                 segmentNodeDict=segmentNodeDict)

        # [1.平均速度3个flim1,f2 f3,2.限速3个，车道数3个，长度 == 10个]
        # 2.2.1  平均车道数，平均限速
        pathAvgSpeedLaneLimSpeed = pathInitFeats.index_select(dim=2, index=torch.LongTensor([1, 2, 3]))
        pathAvgSpeedLaneLimSpeedSum = torch.sum(pathAvgSpeedLaneLimSpeed, dim=1)
        pathLenTensor = torch.tensor(list(pathLen.values())).reshape(num_node_path, 1)
        averagepathLaneLimSpeed = pathAvgSpeedLaneLimSpeedSum / pathLenTensor
        # 2.2.2 获取最大车道数
        pathAvgSpeedLaneLimSpeedLaneNum = pathInitFeats.index_select(dim=2, index=torch.LongTensor([1, 2, 3]))
        pathSpeedLaneLaneNumLimSpeedSumMax = torch.max(pathAvgSpeedLaneLimSpeedLaneNum, dim=1)[0]
        # 2.2.3 获取总长度
        pathLength = pathInitFeats.index_select(dim=2, index=torch.LongTensor([4]))
        pathSumLength = torch.sum(pathLength, dim=1)
        # 2.2.4 获取最小车道数
        pathLaneNumPadInf = pathInitFeatsInf.index_select(dim=2, index=torch.LongTensor([1, 2, 3]))
        pathMinavgSpeedLaneNumLimSpeedMin = torch.min(pathLaneNumPadInf, dim=1)[0]

        onePathFeature = torch.cat(
            [averagepathLaneLimSpeed, pathSpeedLaneLaneNumLimSpeedSumMax, pathMinavgSpeedLaneNumLimSpeedMin,
             pathSumLength],
            dim=1)
        onePathFeat = max_min_scaler.fit_transform(np.array(onePathFeature.float()))
        g_t.nodes['path'].data['onePathFeat'] = torch.tensor(onePathFeat)

        segmentMinMaxScalerFeat = g_t.nodes['segment'].data['feature']
        pathFeat = getPathFeat(numpathnode=num_node_path,
                               PathNodeDict_reverse=PathNodeDict_reverse,
                               max_size=seqmaxsize,
                               segment_feats=segmentMinMaxScalerFeat,
                               segmentNodeDict=segmentNodeDict)
        g_t.nodes['path'].data['pathSegmentFeat'] = pathFeat

        # 3. od特征嵌入
        num_node_od = g_t.num_nodes("od")
        od_count_t = total_od_count_t[t]
        if od_count_t != {}:
            od2path_t_array = getOd2Path_t(t, total_path_count_t, allOd_dict, pathNodeDict)
            od_id = od2path_t_array[0:, 0]
            odNum = torch.zeros(1, 1)
            for i in range(num_node_od):
                if i in od_id:
                    # od i 的数量：od_count_t[allOd_dict_reversed[i]]
                    odNum = torch.vstack((odNum, torch.tensor(od_count_t[allOd_dict_reversed[i]])))
                else:
                    odNum = torch.vstack((odNum, torch.tensor([0.])))
            odNum = odNum[1:]
            g_t.nodes['od'].data['odNum'] = odNum
        else:
            g_t.nodes['od'].data['odNum'] = torch.zeros((num_node_od, 1))

        # 4. 边上信息：path->segment嵌入order信息
        g_t.edges['pass+'].data['orderInfo'] = torch.tensor(df_path2areaOrder['orderInfo']).reshape(-1, 1)

        # 6. 给每个path嵌入path的数量
        path_count_t = total_path_count_t[t]
        if path_count_t != {}:
            od2path_t_array = getOd2Path_t(t, total_path_count_t, allOd_dict, pathNodeDict)
            path_id = od2path_t_array[0:, 1]
            pathNum = torch.zeros(1, 1)
            for i in range(num_node_path):
                if i in path_id:
                    # od i 的数量：od_count_t[allOd_dict_reversed[i]]
                    pathNum = torch.vstack((pathNum, torch.tensor(path_count_t[PathNodeDict_reverse[i]])))
                else:
                    pathNum = torch.vstack((pathNum, torch.tensor([0.])))
            pathNum = pathNum[1:]
            g_t.nodes['path'].data['pathNum'] = pathNum
        else:
            g_t.nodes['path'].data['pathNum'] = torch.zeros((num_node_path, 1))

        save_graphs(f"../HTGWithFeat_sumosy/Hetero_{t}.bin", [g_t])
        print(f'sumosy: {t} is over')


def EmbedFeat_taxibj(startTimestamp, endtimestamps, seqmaxsize, allTimestamps):
    '''
    :param timestamps:
    :param period:
    :param seqmaxsize: 最长path的序列长度
    :return:
    '''
    # 1. 数据读取
    df_flow = pd.read_csv("../data_taxibj/areaRealTimeFlow_taxibj.csv")
    df_avgspeed = pd.read_csv("../data_taxibj/averSpeed_taxibj.csv")
    df_path2areaOrder = pd.read_csv('../data_taxibj/path2area_taxibj.csv')

    total_flow = np.array(df_flow).flatten('C').reshape(-1, 1)
    total_avgspeed = np.array(df_avgspeed).flatten('C').reshape(-1, 1)
    totalsegmentFeat = np.concatenate((total_flow, total_avgspeed), axis=1)
    max_min_scaler = preprocessing.MinMaxScaler()
    minmax_feat = max_min_scaler.fit_transform(totalsegmentFeat)

    with open("../data_taxibj/total_path_count_t_taxibj.txt", 'r') as f:
        total_path_count_t = eval(f.readlines()[0])

    with open("../data_taxibj/total_od_count_t_taxibj.txt") as f:
        total_od_count_t = eval(f.readlines()[0])

    with open("../data_taxibj/odNodeDict_taxibj.txt", "r") as f:
        allOd_dict = eval(f.readlines()[0])
        allOd_dict_reversed = {v: k for k, v in allOd_dict.items()}

    with open("../data_taxibj/pathNodeDict_TaxiBj.txt", "r") as f:
        pathNodeDict = eval(f.readlines()[0])
        PathNodeDict_reverse = {v: k for k, v in pathNodeDict.items()}
        pathLen = {index: len(path) for path, index in pathNodeDict.items()}

    for t in range(startTimestamp, endtimestamps):
        # 2. 读取不带特征的图
        g_t = load_graphs(f"../HtgNoFeat_taxibj/Hetero_{t}.bin", [0])[0][0]
        segment_num = g_t.num_nodes('segment')
        # 1. segment节点嵌入
        # 1.1 segment节点：原始特征

        # 1.2 segment节点：归一化的特征
        g_t.nodes['segment'].data['feature'] = torch.tensor(minmax_feat[t * segment_num:(t + 1) * segment_num]).to(
            torch.float32)
        # 1.3 segment节点：id
        g_t.nodes['segment'].data['id'] = torch.arange(1, segment_num + 1).reshape(-1, 1)

        # 2.path特征嵌入
        segmentId = g_t.nodes['segment'].data['id']
        num_node_path = g_t.num_nodes("path")

        flow = torch.tensor(np.array(df_flow.iloc[t]).reshape(segment_num, 1).astype(float))
        avgspeed = torch.tensor(np.array(df_avgspeed.iloc[t]).reshape(segment_num, 1).astype(float))
        segmentInitFeat = torch.cat([flow, avgspeed], dim=1)

        # 2.1 推演用到的id信息
        segmentIdOfPath = getPathFeat_taxibj(numpathnode=num_node_path,
                                             PathNodeDict_reverse=PathNodeDict_reverse,
                                             max_size=seqmaxsize,
                                             segment_feats=segmentId)
        g_t.nodes['path'].data['segmentId'] = segmentIdOfPath

        # 2.2 路径选择path的信息
        pathInitFeats = getPathFeat_taxibj(numpathnode=num_node_path,
                                           PathNodeDict_reverse=PathNodeDict_reverse,
                                           max_size=seqmaxsize,
                                           segment_feats=segmentInitFeat)

        pathInitFeatsInf = getPathFeatPadWithInf_taxibj(numpathnode=num_node_path,
                                                        PathNodeDict_reverse=PathNodeDict_reverse,
                                                        max_size=seqmaxsize,
                                                        segment_feats=segmentInitFeat)

        # [1.平均速度3个flim1,f2 f3,2.限速3个，车道数3个，长度 == 10个]
        # 2.2.1  平均平均速度
        pathAvgSpeed = pathInitFeats.index_select(dim=2, index=torch.LongTensor([1]))
        pathAvgSpeedSum = torch.sum(pathAvgSpeed, dim=1)
        pathLenTensor = torch.tensor(list(pathLen.values())).reshape(num_node_path, 1)
        pathspeedAvg = pathAvgSpeedSum / pathLenTensor
        # 2.2.2 获取最大平均速度
        pathSpeedSumMax = torch.max(pathAvgSpeed, dim=1)[0]
        # 2.2.4 最小平均速度
        pathAvgSpeedPadInf = pathInitFeatsInf.index_select(dim=2, index=torch.LongTensor([1]))
        pathAvgSpeedMin = torch.min(pathAvgSpeedPadInf, dim=1)[0]

        onePathFeature = torch.cat(
            [pathspeedAvg, pathSpeedSumMax, pathAvgSpeedMin],
            dim=1)
        onePathFeat = max_min_scaler.fit_transform(np.array(onePathFeature.float()))
        g_t.nodes['path'].data['onePathFeat'] = torch.tensor(onePathFeat)

        # 路段特征拼成的路径特征
        segmentMinMaxScalerFeat = g_t.nodes['segment'].data['feature']
        pathFeat = getPathFeat_taxibj(numpathnode=num_node_path,
                                      PathNodeDict_reverse=PathNodeDict_reverse,
                                      max_size=seqmaxsize,
                                      segment_feats=segmentMinMaxScalerFeat)
        g_t.nodes['path'].data['pathSegmentFeat'] = pathFeat

        # 3. od特征嵌入
        num_node_od = g_t.num_nodes("od")
        od_count_t = total_od_count_t[t]
        if od_count_t != {}:
            od2path_t_array = getOd2Path_t(t, total_path_count_t, allOd_dict, pathNodeDict)
            od_id = od2path_t_array[0:, 0]
            odNum = torch.zeros(1, 1)
            for i in range(num_node_od):
                if i in od_id:
                    # od i 的数量：od_count_t[allOd_dict_reversed[i]]
                    odNum = torch.vstack((odNum, torch.tensor(od_count_t[allOd_dict_reversed[i]])))
                else:
                    odNum = torch.vstack((odNum, torch.tensor([0.])))
            odNum = odNum[1:]
            g_t.nodes['od'].data['odNum'] = odNum
        else:
            g_t.nodes['od'].data['odNum'] = torch.zeros((num_node_od, 1))

        # 4. 边上信息：path->segment嵌入order信息
        g_t.edges['pass+'].data['orderInfo'] = torch.tensor(df_path2areaOrder['orderInfo']).reshape(-1, 1)

        # 7. 给每个path嵌入path的数量
        path_count_t = total_path_count_t[t]
        if path_count_t != {}:
            od2path_t_array = getOd2Path_t(t, total_path_count_t, allOd_dict, pathNodeDict)
            path_id = od2path_t_array[0:, 1]
            pathNum = torch.zeros(1, 1)
            for i in range(num_node_path):
                if i in path_id:
                    # od i 的数量：od_count_t[allOd_dict_reversed[i]]
                    pathNum = torch.vstack((pathNum, torch.tensor(path_count_t[PathNodeDict_reverse[i]])))
                else:
                    pathNum = torch.vstack((pathNum, torch.tensor([0.])))
            pathNum = pathNum[1:]
            g_t.nodes['path'].data['pathNum'] = pathNum
        else:
            g_t.nodes['path'].data['pathNum'] = torch.zeros((num_node_path, 1))

        save_graphs(f"../HTGWithFeat_taxibj/Hetero_{t}.bin", [g_t])
        print(t, 'is over')


if __name__ == '__main__':
    seqmaxsize_sumovs = getSeqMaxSize(pathNodeDictFileName="../data_sumovs/pathNodeDict_sumovs.txt")
    EmbedFeat_sumovs(startTimestamp=0,
                     endtimestamps=30 * 24 * 14,
                     seqmaxsize=seqmaxsize_sumovs,
                     allTimestamps=30 * 24 * 14)

    seqmaxsize_sumosy = getSeqMaxSize(pathNodeDictFileName="../data_sumosy/pathNodeDict_sumosy.txt")
    EmbedFeat_sumosy(startTimestamp=0,
                     endtimestamps=30 * 24 * 28,
                     seqmaxsize=seqmaxsize_sumosy,
                     allTimestamps=30 * 24 * 28)

    seqmaxsize_taxibj = getSeqMaxSize(pathNodeDictFileName="../data_taxibj/pathNodeDict_taxibj.txt")
    EmbedFeat_taxibj(startTimestamp=0,
                     endtimestamps=20 * 24 * 7,
                     seqmaxsize=seqmaxsize_taxibj,
                     allTimestamps=20 * 24 * 7)