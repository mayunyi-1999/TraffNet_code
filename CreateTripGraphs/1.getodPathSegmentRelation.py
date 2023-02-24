# -- coding: utf-8 --
# @Time : 2023/2/10 15:06
# @Author : Ma Yunyi
# @File : 1.getodPathSegmentRelation.py
# @Software: PyCharm

import numpy as np
import pandas as pd

def getOdPathSegmentRelation_sumovs():
    with open('../data_sumovs/segmentNodeDict_sumovs.txt', 'r') as f:
        segmentNodeDict = eval(f.readlines()[0])

    with open("../data_sumovs/odNodeDict_sumovs.txt") as f:
        odNodeDict = eval(f.readlines()[0])

    with open("../data_sumovs/pathNodeDict_sumovs.txt") as f:
        pathNodeDict = eval(f.readlines()[0])

    od2path = []
    path2area = []
    for path in list(pathNodeDict.keys()):
        od = (path[0],path[-1])
        od2path.append((odNodeDict[od],pathNodeDict[path]))
        for order,p in enumerate(path):
            path2area.append((pathNodeDict[path],segmentNodeDict[p],order))

    od2path_array = np.array(od2path)
    path2area_array = np.array(path2area)

    df_od2path = pd.DataFrame(od2path_array)
    df_od2path.to_csv("../data_sumovs/od2path_sumovs.csv",header=["connection_from","connection_to"],index=None)

    df_path2area = pd.DataFrame(path2area_array)
    df_path2area.to_csv("../data_sumovs/path2segment_sumovs.csv",header=["connection_from","connection_to","orderInfo"],index=None)


def getOdPathSegmentRelation_sumosy():
    with open('../data_sumosy/segmentNodeDict_sumosy.txt', 'r') as f:
        segmentNodeDict = eval(f.readlines()[0])

    with open("../data_sumosy/odNodeDict_sumosy.txt") as f:
        odNodeDict = eval(f.readlines()[0])

    with open("../data_sumosy/pathNodeDict_sumosy.txt") as f:
        pathNodeDict = eval(f.readlines()[0])

    od2path = []
    path2area = []
    for path in list(pathNodeDict.keys()):
        od = (path[0],path[-1])
        od2path.append((odNodeDict[od],pathNodeDict[path]))
        for order,p in enumerate(path):
            path2area.append((pathNodeDict[path],segmentNodeDict[p],order))

    od2path_array = np.array(od2path)
    path2area_array = np.array(path2area)

    df_od2path = pd.DataFrame(od2path_array)
    df_od2path.to_csv("../data_sumosy/od2path_sumosy.csv",header=["connection_from","connection_to"],index=None)

    df_path2area = pd.DataFrame(path2area_array)
    df_path2area.to_csv("../data_sumosy/path2segment_sumosy.csv",header=["connection_from","connection_to","orderInfo"],index=None)


def getOdPathSegmentRelation_taxibj():
    with open("../data_taxibj/odNodeDict_taxibj.txt") as f:
        odNodeDict = eval(f.readlines()[0])

    with open("../data_taxibj/pathNodeDict_taxibj.txt") as f:
        pathNodeDict = eval(f.readlines()[0])

    pathNode = list(pathNodeDict.keys())

    od2path = []
    for path in pathNode:
        od = (path[0], path[-1])

        od2path.append((odNodeDict[od], pathNodeDict[path]))
    od2path_array = np.array(od2path)

    path2area = []
    for path in pathNode:
        for order, p in enumerate(path):
            path2area.append((pathNodeDict[path], p, order))
    path2area_array = np.array(path2area)

    df_od2path = pd.DataFrame(od2path_array)
    df_od2path.to_csv("../data_taxibj/od2path_taxibj.csv", header=["connection_from", "connection_to"], index=None)

    df_path2area = pd.DataFrame(path2area_array)
    df_path2area.to_csv("../data_taxibj/path2area_taxibj.csv", header=["connection_from", "connection_to", "orderInfo"],
                        index=None)

if __name__ == '__main__':
    # getOdPathSegmentRelation_sumovs()
    getOdPathSegmentRelation_sumosy()
    # getOdPathSegmentRelation_taxibj()