import time

import numpy as np
import torch
import os
from tqdm import tqdm
import sys
from torch.utils.data import DataLoader

from datasetOp.PretrainDataset_sumovs import PreTrainRouteLearningDataset, collect_fn_preSelPath

pathNum = 121
batch_size = 1
window_width = 9
out_len = 1
device = torch.device('cuda:0')

timestampsTrain = 30 * 24 * 8
timestampsVal = 30 * 24 * 2
timestampsTest = 30 * 24 * 4

train_start_idx = 0
val_start_idx = train_start_idx + timestampsTrain  # 60*24*5
test_start_idx = val_start_idx + timestampsVal  # 60*24*6

# train_dataset = PreTrainSelPathDataset(datasetType='Train',
#                                        timestamps=timestampsTrain,
#                                        window_width=window_width, out_len=out_len,
#                                        start_idx=train_start_idx)
# print('trainDataset is ok....')
# train_dataloader = DataLoader(dataset=train_dataset,
#                               batch_size=batch_size,
#                               shuffle=False,
#                               collate_fn=collect_fn_preSelPath)
# print('trainDataLoader is ok....')
#
#
# val_dataset = PreTrainSelPathDataset(datasetType='Val',
#                                      timestamps=timestampsVal,
#                                       window_width=window_width, out_len=out_len,
#                                        start_idx=val_start_idx)
# print('valDataset is ok....')
# val_dataloader = DataLoader(dataset=val_dataset,
#                             batch_size=batch_size,
#                             shuffle=False,
#                             collate_fn=collect_fn_preSelPath)
# print('valDataLoader is ok....')

test_dataset = PreTrainRouteLearningDataset(datasetType='Test',
                                            timestamps=timestampsTest,
                                            window_width=window_width,
                                            out_len=out_len,
                                            start_idx=test_start_idx)
print('TestDataset is ok....')
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=collect_fn_preSelPath)
print('TestDataLoader is ok....')

model = torch.load('pathSelParams_sumovs/pathSel_sumovs.pkl')
print(model)
criterion = torch.nn.MSELoss()
total_loss = 0



test_dataloader = tqdm(test_dataloader, file=sys.stdout)

pathCount = []
pathPredCount = []
pathSelProbList = []
# onePathfeat = []
with torch.no_grad():
    for test_dataloader in test_dataloader:
        batchGraphSeq = test_dataloader[0].to(device)
        labels = test_dataloader[1].to(torch.float32).to(device).reshape(-1, pathNum)

        logits, pathSelProb = model(batchGraphSeq)

        logits = logits.reshape(-1, pathNum)
        pathSelProb = pathSelProb.reshape(-1, pathNum)

        pathPredCount.append(logits)
        pathCount.append(labels)
        pathSelProbList.append(pathSelProb)

        loss = criterion(logits, labels)
        total_loss = total_loss + loss.item()

    total_loss = total_loss / len(test_dataloader)
    epoch_end_time = time.time()

    pathCountNumpy = torch.vstack(pathCount).squeeze(dim=1).cpu().numpy()
    pathPredCountNumpy = torch.vstack(pathPredCount).squeeze(dim=1).cpu().numpy()
    pathSelProbNumpy = torch.vstack(pathSelProbList).squeeze(dim=1).cpu().numpy()
    # onePathfeatNumpy = torch.vstack(onePathfeat).squeeze(dim=1).cpu().numpy()

np.savez('pathCount_pretrain_sumovs.npz', pathCountNumpy)
np.savez('pathPredCount_pretrain_sumovs.npz', pathPredCountNumpy)
np.savez('pathSelProb_pretrain_sumovs.npz', pathSelProbNumpy)
print(f'total_loss:{total_loss}')
