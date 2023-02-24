import time
import torch
import os
from torch.utils.data import DataLoader

from datasetOp.PretrainDataset_sumovs import PreTrainRouteLearningDataset, collect_fn_preSelPath
from model.PreTrainRouteLearningModel_sumovs import PretrainRouteLearningModel
from utils.utils import getSeqMaxSize

window_width = 9
out_len = 1
batch_size = 128
pathNum = 121
epochs = 10000
edgeNum = 254

seqmaxsize = getSeqMaxSize(pathNodeDictFileName="../data/pathNodeDict14daysWhatIf_1step.txt")

device = torch.device('cuda:0')

timestampsTrain = 30 * 24 * 8
timestampsVal = 30 * 24 * 2
timestampsTest = 30 * 24 * 4

train_start_idx = 0
val_start_idx = train_start_idx + timestampsTrain
test_start_idx = val_start_idx + timestampsVal

model = PretrainRouteLearningModel(seq_max_len=seqmaxsize,
                                   edge_num=edgeNum).to(device)
print(model)
# criterion_old = torch.nn.MSELoss()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_dataset = PreTrainRouteLearningDataset(datasetType='Train',
                                             timestamps=timestampsTrain,
                                             window_width=window_width, out_len=out_len,
                                             start_idx=train_start_idx)
print('trainDataset is ok....')
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collect_fn_preSelPath)
print('trainDataLoader is ok....')

val_dataset = PreTrainRouteLearningDataset(datasetType='Val',
                                           timestamps=timestampsVal,
                                           window_width=window_width,
                                           out_len=out_len,
                                           start_idx=val_start_idx)
print('valDataset is ok....')
val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collect_fn_preSelPath)
print('valDataLoader is ok....')

start_epoch = 0
min_val_total_loss = 100000000
for epoch in range(epochs):
    epoch_start_time = time.time()
    # train model......
    train_total_loss = 0
    model.train()
    for i, data in enumerate(train_dataloader):
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        batchGraphSeq = data[0].to(device)
        labels = data[1].to(torch.float32).to(device)

        logits, _ = model(batchGraphSeq)

        loss = criterion(logits, labels)
        train_total_loss = train_total_loss + loss.item()
        loss.backward()
        optimizer.step()

    train_loss = train_total_loss / len(train_dataloader)

    # validate model....
    model.eval()
    val_total_loss = 0

    with torch.no_grad():
        for val_data in val_dataloader:
            batchGraphSeq = val_data[0].to(device)
            labels = val_data[1].to(torch.float32).to(device)

            logits, _ = model(batchGraphSeq)

            loss = criterion(logits, labels)
            val_total_loss = val_total_loss + loss.item()

    val_loss = val_total_loss / len(val_dataloader)
    epoch_end_time = time.time()
    if not os.path.exists('pathSelParams_sumovs'):
        os.makedirs('pathSelParams_sumosv')
    with open('pathSelParams_sumosv/loss_sumovs.txt', 'a') as f:
        f.write(
            f"[epoch:{(epoch + start_epoch)} | train_total_loss:{train_total_loss},val_total_loss:{val_total_loss} | avgbatchTrainLoss:{train_loss},avgbatchValLoss:{val_loss} | time:{epoch_end_time - epoch_start_time}" + '\n')
    print(
        f"[epoch:{epoch + start_epoch} | train_total_loss:{train_total_loss},val_total_loss:{val_total_loss} | avgbatchTrainLoss:{train_loss},avgbatchValLoss:{val_loss} | time:{epoch_end_time - epoch_start_time}")

    if min_val_total_loss > val_total_loss:
        torch.save(model, f'pathSelParams_sumovs/pathSel_sumovs.pkl')
        min_val_total_loss = val_total_loss

