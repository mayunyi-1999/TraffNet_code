import torch
from torch.utils.data import DataLoader
import os
import warnings
import pathlib

from datasetOp.dataset_taxibj import DataSetTaxiBJ, traffNet_collect_fn
from utils.eval_utils import evaluateTraff_taxibj
from utils.utils import getSeqMaxSize

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings("ignore")

window_width = 3
batch_size = 1
edgeNum = 81
horizon = 7  # 预测1个时刻

# 训练集，测试集，验证集的长度
timestampsTrain = 20 * 24 * 4
timestampsVal = 20 * 24 * 1
timestampsTest = 20 * 24 * 2

# 训练集，测试集，验证集开始的索引
train_start_idx = 0
val_start_idx = train_start_idx + timestampsTrain  # 60*24*5
test_start_idx = val_start_idx + timestampsVal  # 60*24*6

folder = pathlib.Path(__file__).parent.parent.resolve()
seq_max_size = getSeqMaxSize(os.path.join(folder, "data/pathNodeDict_TaxiBj.txt"))

device = torch.device('cuda:0')
# numpathnode = len(pathNodeDict)

criterion = torch.nn.MSELoss()

train_dataset = DataSetTaxiBJ(datasetType='Train',
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

test_dataset = DataSetTaxiBJ(datasetType='Test',
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

model = torch.load('../results/traffNet_taxibj/model_taxibj.pkl')
print(model)

evaluateTraff_taxibj(model=model,
                     dataloader=test_dataloader,
                     modelName='HTG',
                     edgeNum=edgeNum,
                     device=device,
                     out_len=horizon)
