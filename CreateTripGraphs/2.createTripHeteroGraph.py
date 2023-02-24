# -- coding: utf-8 --
# @Time : 2023/2/10 15:06
# @Author : Ma Yunyi
# @Software: PyCharm

from dgl import save_graphs
import dgl
import pandas as pd
import torch

def getHeteroRelation(relationfile):
    df = pd.read_csv(relationfile)
    connection_from = df['connection_from']
    connection_to = df['connection_to']

    from_tensor = torch.tensor(connection_from.tolist())
    to_tensor = torch.tensor(connection_to.tolist())
    return from_tensor, to_tensor


def createTrafficHeteroGraph(od_path_from, od_path_to, path_segment_from, path_segment_to, segment_segment_from, segment_segment_to):
    G = dgl.heterograph({
        ('od', 'select+', 'path'): (od_path_from, od_path_to),
        ('path', 'select-', 'od'): (od_path_to, od_path_from),
        ('path', 'pass+', 'segment'): (path_segment_from, path_segment_to),
        ('segment', 'pass-', 'path'): (path_segment_to, path_segment_from),
        ('segment', 'connect+', 'segment'): (segment_segment_from, segment_segment_to),
        ('segment', 'connect-', 'segment'): (segment_segment_to, segment_segment_from)
    })
    return G

def getHeteroTripGraph_sumovs():
    od_path_from,od_path_to = getHeteroRelation('../data_sumovs/od2path_sumovs.csv')
    path_segment_from, path_segment_to = getHeteroRelation('../data_sumovs/path2segment_sumovs.csv')
    segment_segment_from,segment_segment_to = getHeteroRelation('../data_sumovs/neighbor_sumovs.csv')
    G = createTrafficHeteroGraph(od_path_from,od_path_to,path_segment_from,path_segment_to,segment_segment_from,segment_segment_to)
    save_graphs("../HeteroFeat_sumovs/Hetero.bin", [G])

def getHeteroTripGraph_sumosy():
    od_path_from,od_path_to = getHeteroRelation('../data_sumosy/od2path_sumosy.csv')
    path_segment_from, path_segment_to = getHeteroRelation('../data_sumosy/path2segment_sumosy.csv')
    segment_segment_from,segment_segment_to = getHeteroRelation('../data_sumosy/neighbor_sumosy.csv')
    G = createTrafficHeteroGraph(od_path_from,od_path_to,path_segment_from,path_segment_to,segment_segment_from,segment_segment_to)
    save_graphs("../HeteroFeat_sumosy/Hetero.bin", [G])

def getHeteroTripGraph_taxibj():
    od_path_from,od_path_to = getHeteroRelation('../data_taxibj/od2path_taxibj.csv')
    path_segment_from, path_segment_to = getHeteroRelation('../data_taxibj/path2area_taxibj.csv')
    segment_segment_from,segment_segment_to = getHeteroRelation('../data_taxibj/neighbor_taxibj.csv')
    G = createTrafficHeteroGraph(od_path_from,od_path_to,path_segment_from,path_segment_to,segment_segment_from,segment_segment_to)
    save_graphs("../HeteroFeat_taxibj/Hetero.bin", [G])





if __name__ == '__main__':
    getHeteroTripGraph_sumovs()
    print('sumo_vs is ok...')

    getHeteroTripGraph_sumosy()
    print('sumo_sy is ok...')

    getHeteroTripGraph_taxibj()
    print('taxi_bj is ok...')