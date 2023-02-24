# -- coding: utf-8 --
# @Time : 2023/2/10
# @Author : Ma Yunyi
# @Software: PyCharm

import torch
from torch.utils.data import DataLoader
import os
import warnings
import pathlib

from datasetOp.dataset_sumovs import traffNet_collect_fn, DataSetSumoVS
from utils.eval_utils import evaluateTraffNet
from utils.utils import getSeqMaxSize

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings("ignore")

window_width = 9
batch_size = 1
edgeNum = 254
horizon = 1  # 预测1个时刻

# 训练集，测试集，验证集的长度
timestampsTrain = 30 * 24 * 8
timestampsVal = 30 * 24 * 2
timestampsTest = 30 * 24 * 4

# 训练集，测试集，验证集开始的索引
train_start_idx = 0
val_start_idx = train_start_idx + timestampsTrain  # 60*24*5
test_start_idx = val_start_idx + timestampsVal  # 60*24*6

folder = pathlib.Path(__file__).parent.parent.resolve()
seq_max_size = getSeqMaxSize(os.path.join(folder, "data_sumovs/pathNodeDict14daysWhatIf_1step.txt"))


criterion = torch.nn.MSELoss()

train_dataset = DataSetSumoVS(datasetType='Train',
                                               timestamps=timestampsTrain,
                                               window_width=window_width,
                                               horizon=horizon,
                                               start_idx=train_start_idx)
print('trainDataset is ok....')
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=traffNet_collect_fn)
print('trainDataLoader is ok....')

test_dataset = DataSetSumoVS(datasetType='Test',
                                              timestamps=timestampsTest,
                                              window_width=window_width,
                                              horizon=horizon,
                                              start_idx=test_start_idx)
print('testDataset is ok....')
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=traffNet_collect_fn)
print('testDataLoader is ok....')

model = torch.load('../results/traffNet_whatif_1_0214/whatif_horizon1_0214_2.pkl')
print(model)


evaluateTraffNet(model=model,
                 dataloader=test_dataloader,
                 modelName='HTG',
                 edgeNum=edgeNum)
