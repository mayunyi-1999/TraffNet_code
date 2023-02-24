import dgl
import pandas as pd
import numpy as np


def getLabel(t, window_width):
    areaflow = pd.read_csv("../data_sumosy/realflowAll28days_sumosy.csv")
    label = areaflow.iloc[t + window_width].values
    return label


def getBatchGraph(t, window_width):
    indat = []
    for i in range(t, t + window_width):
        dat_i_G = dgl.load_graphs(f'../HTGWithFeat_sumosy/Hetero_{i}.bin', [0])[0][0]
        indat.append(dat_i_G)
    graphSeqATimeWindow = dgl.batch(indat)  # 6个图构成的图序列
    dgl.save_graphs(f"../batchHtgGraph_sumosy/batchHtg{t}.bin", [graphSeqATimeWindow])


def getGraphLabels(start_timestamps, end_timestamps, window_width, out_len, edgeNum):
    labels = np.zeros((out_len, edgeNum))
    for t in range(start_timestamps, end_timestamps):
        label = getLabel(t, window_width)
        getBatchGraph(t, window_width)  # 存储图
        labels = np.vstack((labels, label))
        print(f'{t} is ok....')

    labels = labels.reshape(-1, out_len, edgeNum)
    label2Mins = labels[1:]
    return label2Mins


if __name__ == '__main__':
    window_width = 9
    out_len = 1
    edgeNum = 254
    start_timestamps, end_timestamps = 0, 30*24*28-window_width
    # start_timestamps, end_timestamps = 0, 4000
    label2Mins = getGraphLabels(start_timestamps=start_timestamps,
                                end_timestamps=end_timestamps,
                                window_width=window_width,
                                out_len=out_len,
                                edgeNum=edgeNum)
    np.savez(f'../data_sumosy/labels_sumosy.npz', label2Mins)
    print('ok')
