import time
import torch
import os
from tqdm import tqdm
import sys
from torch.utils.data import DataLoader
import pathlib

from datasetOp.dataset_sumosy import DataSetSumoSy, traffNet_collect_fn
from model.PredTraffModel_sumosy import PredTraffModel_sumosy
from utils.utils import getSeqMaxSize

# 加载模型超参数

folder = pathlib.Path(__file__).parent.parent.resolve()
seq_max_size = getSeqMaxSize(os.path.join(folder, "data/pathNodeDict_sumosy.txt"))

window_width = 9
batch_size = 10
edgeNum = 254
horizon = 7
lr = 0.00005
epochs = 10000
# device = torch.device('cuda:0')

# 训练集，测试集，验证集的长度
timestampsTrain = 30 * 24 * 16
timestampsVal = 30 * 24 * 4
timestampsTest = 30 * 24 * 8
# 训练集，测试集，验证集开始的索引
train_start_idx = 0
val_start_idx = train_start_idx + timestampsTrain  # 60*24*5
test_start_idx = val_start_idx + timestampsVal  # 60*24*6

# 加载数据集
train_dataset = DataSetSumoSy(datasetType='Train',
                              timestamps=timestampsTrain,
                              window_width=window_width,
                              horizon=horizon,
                              start_idx=train_start_idx)
print('trainDataset is ok....')
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=traffNet_collect_fn)
print('trainDataLoader is ok....')

val_dataset = DataSetSumoSy(datasetType='Val',
                            timestamps=timestampsVal,
                            window_width=window_width,
                            horizon=horizon,
                            start_idx=val_start_idx)
print('valDataset is ok....')
val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=traffNet_collect_fn)

# 定义模型，目标函数，优化器
start_epoch = 0
model = PredTraffModel_sumosy(seq_max_len=seq_max_size,
                              window_width=window_width,
                              edge_num=edgeNum,
                              batch_size=batch_size,
                              horizon=horizon)
print(model)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 加载预训练参数
weights_path = 'pathSelParams_sumosy.pth'
pre_weights_dict = torch.load(weights_path, map_location=torch.device('cuda:0'))
missing_keys, unexpected_keys = model.load_state_dict(pre_weights_dict, strict=False)
print(f'missing_key:{missing_keys}')
print(f'unexpected_key:{unexpected_keys}')

print(f'batch_size:{batch_size}')
# 开始训练
min_val_total_loss = 10000
for epoch in range(epochs):
    epoch_start_time = time.time()
    # train model......
    train_total_loss = 0
    train_dataloader = tqdm(train_dataloader, file=sys.stdout)

    model.train()
    for i, data in enumerate(train_dataloader):
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        batchGraphSeq = data[0].to(torch.device('cuda:0'))
        labels = data[1].to(torch.float32).to(torch.device('cuda:0'))
        logits, _, _ = model(batchGraphSeq)

        loss = criterion(logits, labels)

        train_total_loss = train_total_loss + loss.item()

        loss.backward()
        optimizer.step()

    train_loss = train_total_loss / len(train_dataloader)

    # validate model....
    model.eval()

    val_total_loss = 0
    val_dataloader = tqdm(val_dataloader, file=sys.stdout)

    with torch.no_grad():
        for val_data in val_dataloader:
            batchGraphSeq = val_data[0].to(torch.device('cuda:0'))
            labels = val_data[1].to(torch.float32).to(torch.device('cuda:0'))

            logits, _, _ = model(batchGraphSeq)

            loss = criterion(logits, labels)
            val_total_loss = val_total_loss + loss.item()

    val_loss = val_total_loss / len(val_dataloader)
    epoch_end_time = time.time()
    if not os.path.exists(f'../results/traffNet_sumosy'):
        os.makedirs(f'../results/traffNet_sumosy')
    with open(f'../results/traffNet_sumosy/loss_whatif_horizon{horizon}_0222_1.txt', 'a') as f:
        f.write(
            f"[epoch:{(epoch + start_epoch)} | train_total_loss:{train_total_loss},val_total_loss:{val_total_loss} | avgbatchTrainLoss:{train_loss},avgbatchValLoss:{val_loss} | time:{epoch_end_time - epoch_start_time}" + '\n')
    print(
        f"[epoch:{epoch + start_epoch} | train_total_loss:{train_total_loss},val_total_loss:{val_total_loss} | avgbatchTrainLoss:{train_loss},avgbatchValLoss:{val_loss} | time:{epoch_end_time - epoch_start_time}")

    if min_val_total_loss > val_total_loss:
        torch.save(model, f'../results/traffNet_sumosy/whatif_horizon{horizon}_0222_1.pkl')
        min_val_total_loss = val_total_loss
