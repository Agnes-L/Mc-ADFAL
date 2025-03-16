# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import math
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import argparse

from metrics import get_cindex
from dataset import *
from model import MGraphDTA
from utils import *
from log.train_logger import TrainLogger
from sklearn.metrics import r2_score

def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()
    running_cindex = AverageMeter()
    running_r2score = AverageMeter()

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            cindex = get_cindex(data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))
            r2score = r2_score(data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))  # 添加 R2 计算

            label = data.y
            running_loss.update(loss.item(), label.size(0))
            running_cindex.update(cindex, data.y.size(0))
            running_r2score.update(r2score, data.y.size(0))  # 更新 R2

    epoch_loss = running_loss.get_average()
    epoch_cindex = running_cindex.get_average()
    #epoch_rmse = np.sqrt(epoch_loss)
    epoch_r2score = running_r2score.get_average()  # 获取平均 R2

    running_loss.reset()
    running_cindex.reset()
    running_r2score.reset()  # 重置 R2

    model.train()

    return epoch_loss, epoch_cindex, epoch_r2score

# 定义带权重系数的均方误差损失函数
def weighted_mse_loss(predicted, target, weight_compound=1.0, weight_protein=367):
    mse_loss = nn.functional.mse_loss(predicted, target)
    mse_loss_compound = (mse_loss * weight_compound) / (weight_compound + weight_protein)
    mse_loss_protein = (mse_loss * weight_protein) / (weight_compound + weight_protein)
    sum_weighted_loss = (mse_loss_compound * weight_compound) + (mse_loss_protein * weight_protein)

    return sum_weighted_loss

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--dataset', required=True, help='davis or kiba')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    args = parser.parse_args()

    params = dict(
        data_root="data",
        save_dir="save",
        dataset=args.dataset,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size
    )

    logger = TrainLogger(params)
    logger.info(__file__)

    DATASET = params.get("dataset")
    save_model = params.get("save_model")
    data_root = params.get("data_root")
    fpath = os.path.join(data_root, DATASET)

    train_set = GNNDataset(fpath, types='train')
    val_set = GNNDataset(fpath, types='val')
    test_set = GNNDataset(fpath, types='test')

    logger.info(f"Number of train: {len(train_set)}")
    logger.info(f"Number of val: {len(val_set)}")
    logger.info(f"Number of test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=params['batch_size'], shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=params['batch_size'], shuffle=False, num_workers=8)


    device = torch.device('cuda:0')
    model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)

    epochs = 3000
    steps_per_epoch = 50
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    break_flag = False

    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    #criterion = weighted_mse_loss
    criterion = nn.MSELoss()

    global_step = 0
    global_epoch = 0
    early_stop_epoch = 400

    running_loss = AverageMeter()
    running_cindex = AverageMeter()
    #running_rmse = AverageMeter()
    running_r2score = AverageMeter() 
    running_best_mse = BestMeter("min")

    model.train()

    for i in range(num_iter):
        if break_flag:
            break

        for data in train_loader:

            global_step += 1       
            data = data.to(device)
            pred = model(data)

            loss = weighted_mse_loss(pred.view(-1).squeeze(), data.y.view(-1).squeeze())
            cindex = get_cindex(data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))
            r2score = r2_score(data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))  # 添加 R2 计算
            torch.Tensor.cpu(loss)
            #rmse = np.sqrt(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), data.y.size(0)) 
            running_cindex.update(cindex, data.y.size(0))
            running_r2score.update(r2score, data.y.size(0))  # 更新 R2
            #running_rmse.update(rmse, data.y.size(0))
			
			

            if global_step % steps_per_epoch == 0:

                global_epoch += 1

                epoch_loss = running_loss.get_average()
                epoch_cindex = running_cindex.get_average()
                #epoch_rmse = np.sqrt(epoch_loss)
                epoch_r2score = running_r2score.get_average()  # 获取平均 R2

				
                running_loss.reset()
                running_cindex.reset()
                running_r2score.reset()  # 重置 R2
                #running_rmse.reset()

                val_loss, val_cindex, val_r2score = val(model, criterion, val_loader, device)
                test_loss, test_cindex, test_r2score = val(model, criterion, test_loader, device)

                msg = "epoch-%d, loss-%.4f, cindex-%.4f, r2score-%.4f, val_cindex-%.4f, val_loss-%.4f, val_r2score-%.4f, test_cindex-%.4f, test_loss-%.4f, test_r2score-%.4f" % (global_epoch, epoch_loss, epoch_cindex, epoch_r2score, val_cindex, val_loss, val_r2score, test_cindex, test_loss, test_r2score)
                logger.info(msg)

                if test_loss < running_best_mse.get_best():
                    running_best_mse.update(test_loss)
                    if save_model:
                        save_model_dict(model, logger.get_model_dir(), msg)
        
                else:
                    count = running_best_mse.counter()
                    if count > early_stop_epoch:
                        logger.info(f"early stop in epoch {global_epoch}")
                        break_flag = True
                        break

if __name__ == "__main__":
    main()
