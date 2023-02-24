# -- coding: utf-8 --
# @Time : 2023/2/10 15:06
# @Author : Ma Yunyi
# @Software: PyCharm

from dgl import load_graphs, save_graphs


def getGraphSeq_sumovs(timestamps):
    g = load_graphs(f"../HeteroFeat_sumovs/Hetero.bin", [0])[0][0]
    for t in range(timestamps):
        save_graphs(f"../HtgNoFeat_sumovs/Hetero_{t}.bin", [g])
        print(f'sumovs: {t} is ok')


def getGraphSeq_sumosy(timestamps):
    g = load_graphs(f"../HeteroFeat_sumosy/Hetero.bin", [0])[0][0]
    for t in range(timestamps):
        save_graphs(f"../HtgNoFeat_sumosy/Hetero_{t}.bin", [g])
        print(f'sumosy: {t} is ok')


def getGraphSeq_taxibj(timestamps):
    g = load_graphs(f"../HeteroFeat_taxibj/Hetero.bin", [0])[0][0]
    for t in range(timestamps):
        save_graphs(f"../HtgNoFeat_taxibj/Hetero_{t}.bin", [g])
        print(f'taxibj: {t} is ok')


if __name__ == '__main__':
    getGraphSeq_sumovs(timestamps=30 * 24 * 14)
    getGraphSeq_sumosy(timestamps=30 * 24 * 28)
    getGraphSeq_taxibj(timestamps=20 * 24 * 7)
