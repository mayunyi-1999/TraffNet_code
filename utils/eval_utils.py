import torch
import numpy as np
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def evaluateTraffNet(dataloader, model,edgeNum: int, modelName: str):
    allLogit = []
    allLabels = []
    with torch.no_grad():
        dataloader = tqdm(dataloader, file=sys.stdout)
        for i, data in enumerate(dataloader):
            torch.cuda.empty_cache()
            batchGraphSeq = data[0].to(torch.device('cuda:0'))
            labels = data[1].to(torch.float32).to(torch.device('cuda:0'))
            logits, predFlow, pathFlow = model(batchGraphSeq)

            logits[logits<0] = 0
            allLabels.append(labels)
            allLogit.append(logits)

    allLogitNumpy = torch.vstack(allLogit).squeeze(dim=1).cpu().numpy().reshape(-1, edgeNum)
    allLabelsNumpy = torch.vstack(allLabels).squeeze(dim=1).cpu().numpy().reshape(-1, edgeNum)

    mae = mean_absolute_error(allLabelsNumpy, allLogitNumpy)
    rmse = np.sqrt(mean_squared_error(allLabelsNumpy, allLogitNumpy))

    print(modelName + f":   RMSE:{rmse},MAE:{mae}")



def evaluateTraff_sumosy(dataloader, model, edgeNum: int, modelName: str,out_len):
    '''
        --------Trainset----------00
    '''
    allLogit = []
    allLabels = []
    with torch.no_grad():
        dataloader = tqdm(dataloader, file=sys.stdout)
        for i, data in enumerate(dataloader):
            batchGraphSeq = data[0].to(torch.device('cuda:0'))
            labels = data[1].to(torch.float32).to(torch.device('cuda:0'))
            logits,_,_ = model(batchGraphSeq)
            logits[logits < 0] = 0
            allLabels.append(labels)
            allLogit.append(logits)


    allLogitNumpy = torch.vstack(allLogit).squeeze(dim=1).cpu().numpy()
    allLabelsNumpy = torch.vstack(allLabels).squeeze(dim=1).cpu().numpy()


    mae_h1 = mean_absolute_error(allLabelsNumpy[:, 0, :], allLogitNumpy[:, 0, :])
    mae_h2 = mean_absolute_error(allLabelsNumpy[:, 1, :], allLogitNumpy[:, 1, :])
    mae_h3 = mean_absolute_error(allLabelsNumpy[:, 2, :], allLogitNumpy[:, 2, :])
    mae_h4 = mean_absolute_error(allLabelsNumpy[:, 3, :], allLogitNumpy[:, 3, :])
    mae_h5 = mean_absolute_error(allLabelsNumpy[:, 4, :], allLogitNumpy[:, 4, :])
    mae_h6 = mean_absolute_error(allLabelsNumpy[:, 5, :], allLogitNumpy[:, 5, :])
    mae_h7 = mean_absolute_error(allLabelsNumpy[:, 6, :], allLogitNumpy[:, 6, :])

    rmse_h1 = np.sqrt(mean_squared_error(allLabelsNumpy[:, 0, :], allLogitNumpy[:, 0, :]))
    rmse_h2 = np.sqrt(mean_squared_error(allLabelsNumpy[:, 1, :], allLogitNumpy[:, 1, :]))
    rmse_h3 = np.sqrt(mean_squared_error(allLabelsNumpy[:, 2, :], allLogitNumpy[:, 2, :]))
    rmse_h4 = np.sqrt(mean_squared_error(allLabelsNumpy[:, 3, :], allLogitNumpy[:, 3, :]))
    rmse_h5 = np.sqrt(mean_squared_error(allLabelsNumpy[:, 4, :], allLogitNumpy[:, 4, :]))
    rmse_h6 = np.sqrt(mean_squared_error(allLabelsNumpy[:, 5, :], allLogitNumpy[:, 5, :]))
    rmse_h7 = np.sqrt(mean_squared_error(allLabelsNumpy[:, 6, :], allLogitNumpy[:, 6, :]))


    print(f'rmse_h1:{rmse_h1}  mae_h1:{mae_h1}')
    print(f'rmse_h2:{rmse_h2}  mae_h2:{mae_h2}')
    print(f'rmse_h3:{rmse_h3}  mae_h3:{mae_h3}')
    print(f'rmse_h4:{rmse_h4}  mae_h4:{mae_h4}')
    print(f'rmse_h5:{rmse_h5}  mae_h5:{mae_h5}')
    print(f'rmse_h6:{rmse_h6}  mae_h6:{mae_h6}')
    print(f'rmse_h7:{rmse_h7}  mae_h7:{mae_h7}')

    allLogitNumpy = allLogitNumpy.reshape(-1, edgeNum * out_len)
    allLabelsNumpy = allLabelsNumpy.reshape(-1, edgeNum * out_len)

    mae = mean_absolute_error(allLabelsNumpy, allLogitNumpy)
    rmse = np.sqrt(mean_squared_error(allLabelsNumpy, allLogitNumpy))

    print(modelName + f":   RMSE:{rmse},MAE:{mae}")


def evaluateTraff_taxibj(dataloader, model, edgeNum: int,
                              modelName: str, device,out_len):
    '''
        --------Trainset----------00
    '''
    allLogit = []
    allLabels = []
    with torch.no_grad():
        dataloader = tqdm(dataloader, file=sys.stdout)
        for i, data in enumerate(dataloader):
            batchGraphSeq = data[0].to(device)
            labels = data[1].to(torch.float32).to(device)
            logits,_,_ = model(batchGraphSeq)
            logits[logits < 0] = 0
            allLabels.append(labels)
            allLogit.append(logits)


    allLogitNumpy = torch.vstack(allLogit).squeeze(dim=1).cpu().numpy()
    allLabelsNumpy = torch.vstack(allLabels).squeeze(dim=1).cpu().numpy()


    mae_h1 = mean_absolute_error(allLabelsNumpy[:, 0, :], allLogitNumpy[:, 0, :])
    mae_h2 = mean_absolute_error(allLabelsNumpy[:, 1, :], allLogitNumpy[:, 1, :])
    mae_h3 = mean_absolute_error(allLabelsNumpy[:, 2, :], allLogitNumpy[:, 2, :])
    mae_h4 = mean_absolute_error(allLabelsNumpy[:, 3, :], allLogitNumpy[:, 3, :])
    mae_h5 = mean_absolute_error(allLabelsNumpy[:, 4, :], allLogitNumpy[:, 4, :])
    mae_h6 = mean_absolute_error(allLabelsNumpy[:, 5, :], allLogitNumpy[:, 5, :])
    mae_h7 = mean_absolute_error(allLabelsNumpy[:, 6, :], allLogitNumpy[:, 6, :])

    rmse_h1 = np.sqrt(mean_squared_error(allLabelsNumpy[:, 0, :], allLogitNumpy[:, 0, :]))
    rmse_h2 = np.sqrt(mean_squared_error(allLabelsNumpy[:, 1, :], allLogitNumpy[:, 1, :]))
    rmse_h3 = np.sqrt(mean_squared_error(allLabelsNumpy[:, 2, :], allLogitNumpy[:, 2, :]))
    rmse_h4 = np.sqrt(mean_squared_error(allLabelsNumpy[:, 3, :], allLogitNumpy[:, 3, :]))
    rmse_h5 = np.sqrt(mean_squared_error(allLabelsNumpy[:, 4, :], allLogitNumpy[:, 4, :]))
    rmse_h6 = np.sqrt(mean_squared_error(allLabelsNumpy[:, 5, :], allLogitNumpy[:, 5, :]))
    rmse_h7 = np.sqrt(mean_squared_error(allLabelsNumpy[:, 6, :], allLogitNumpy[:, 6, :]))


    print(f'rmse_h1:{rmse_h1}  mae_h1:{mae_h1}')
    print(f'rmse_h2:{rmse_h2}  mae_h2:{mae_h2}')
    print(f'rmse_h3:{rmse_h3}  mae_h3:{mae_h3}')
    print(f'rmse_h4:{rmse_h4}  mae_h1:{mae_h4}')
    print(f'rmse_h5:{rmse_h5}  mae_h2:{mae_h5}')
    print(f'rmse_h6:{rmse_h6}  mae_h3:{mae_h6}')
    print(f'rmse_h7:{rmse_h7}  mae_h1:{mae_h7}')

    allLogitNumpy = allLogitNumpy.reshape(-1, edgeNum * out_len)
    allLabelsNumpy = allLabelsNumpy.reshape(-1, edgeNum * out_len)

    mae = mean_absolute_error(allLabelsNumpy, allLogitNumpy)
    rmse = np.sqrt(mean_squared_error(allLabelsNumpy, allLogitNumpy))

    print(modelName + f":   RMSE:{rmse},MAE:{mae}")