from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import Tensor
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
import cvae
import utils
import dataloader
import matplotlib.pyplot as plt

import os
import time
import random

print(torch.__version__)

cfg = {
    'data_path': '/datasets/ETH/seq_eth/',
    'model_params': {
        'model_cnn': "resnet34",
        'scene_weight': 640,
        'scene_height': 480,
        'history_num_frames': 8,
        'history_delta_time': 0.25,
        'future_num_frames': 12,
        'model_name': "CVAE",
        'lr': 2e-3,
        'weight_path': '',
        'train': True,
        'predict': False,
    },
    'train_data_loader': {
        'batch_size': 12,
        'shuffle': True,
        'num_workers': 4,
    },
    'valid_data_loader': {
        'batch_size': 12,
        'shuffle': True,
        'num_workers': 4,
    },
    'test_data_loader': {
        'batch_size': 128,
        'shuffle': False,
        'num_workers': 4,
    },
    'train_params': {
        'epoch': 20,
        'checkpoint_steps': 100,
        'valid_steps': 10,
    }
}

def forward(scene, his_traj, targets, model, device, criterion=cvae.loss_cvae, compute_loss=True):
    # Forward pass
    preds, confidences, context, mean, var = model(scene, his_traj)
    # skip compute loss if we are doing prediction
    loss, loss_ade = criterion(targets, preds, confidences, mean, var) if compute_loss else 0
    return loss, loss_ade, preds, confidences


if __name__ == '__main__':
    # 加载数据集，准备device
    DIR_INPUT = cfg["data_path"]
    os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT

    train_cfg = cfg["train_data_loader"]
    train_dataset = dataloader.EthSceneDataset(os.path.join(cfg['data_path'], 'train'))
    train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"],
                                  batch_size=train_cfg["batch_size"], num_workers=train_cfg["num_workers"],
                                  drop_last=True)

    valid_cfg = cfg["valid_data_loader"]
    valid_dataset = dataloader.EthSceneDataset(os.path.join(cfg['data_path'], 'valid'))
    valid_dataloader = DataLoader(valid_dataset, shuffle=valid_cfg["shuffle"], batch_size=valid_cfg["batch_size"],
                                  num_workers=valid_cfg["num_workers"], drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    train_writer = SummaryWriter('../log/eth/cvae', comment='cvae')

    # 建立模型
    model = cvae.CVAE(cnn_model=cfg["model_params"]["model_cnn"],
                      weight=cfg['model_params']['scene_weight'],
                      height=cfg['model_params']['scene_height'],
                      channels=3, cont_dim=256)

    # load weight if there is a pretrained model
    weight_path = cfg["model_params"]["weight_path"]
    if weight_path:
        model.load_state_dict(torch.load(weight_path))
        print(weight_path, "loaded")

    model.to(device)
    learning_rate = cfg["model_params"]["lr"]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    print(f'device {device}')
    # raster_size = (640, 480)
    # summary(model, input_size=raster_size)
    # print(model)
    torch.backends.cudnn.benchmark = True

    # 开始训练
    if cfg["model_params"]["train"]:
        tr_it = iter(train_dataloader)
        tr_it_valid = iter(valid_dataloader)
        epochs = cfg["train_params"]["epoch"]
        progress_bar = tqdm(range(1, len(train_dataloader)), mininterval=5.)
        losses = []
        iterations = []
        metrics = []
        times = []
        model_name = cfg["model_params"]["model_name"]
        checkpoint_steps = cfg['train_params']['checkpoint_steps']
        valid_steps = cfg['train_params']['valid_steps']
        t_start = time.time()
        torch.set_grad_enabled(True)
        i = 0
        for epoch_i in range(epochs):

            for _ in progress_bar:
                try:
                    data = next(tr_it)
                except StopIteration:
                    tr_it = iter(train_dataloader)
                    data = next(tr_it)

                model.train()
                scene = data[0].to(device)
                scene = scene.permute(0, 3, 2, 1)
                his_traj = data[1].to(device)
                his_traj = his_traj.permute(1, 0, 2)
                targets = data[2].to(device)
                loss, _, _, _ = forward(scene.float(), his_traj.float(), targets.float(), model, device)
                # Backward pass
                scheduler.step()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss = loss.item()
                losses.append(loss)
                i += 1
                train_writer.add_scalar('loss', loss, i)
                if i % checkpoint_steps == 0:
                    mean_losses = np.mean(losses)
                    losses = []
                    train_writer.add_scalar('mean_losses', mean_losses, i)
                    timespent = (time.time() - t_start) / 60
                    curr_lr = optimizer.param_groups[0]['lr']
                    train_writer.add_scalar('lr', curr_lr, i)
                    print('epoch: %5d' % epoch_i, 'i: %5d' % i,
                          'loss(avg): %10.5f' % mean_losses, 'lr(curr): %f' % curr_lr,
                          ' %.2fmins' % timespent, end=' | \n')
                    checkpoint = {"model_state_dict": model.state_dict(),
                                  "optimizer_state_dict": optimizer.state_dict(),
                                  "epoch": i}
                    path_checkpoint = "./model/checkpoint_{}_epoch.pkl".format(i)
                    torch.save(checkpoint, path_checkpoint)
                    iterations.append(i)
                    metrics.append(mean_losses)
                    times.append(timespent)

            if epoch_i % valid_steps == 0:
                model.eval()
                valid_ades = []
                valid_fdes = []
                torch.set_grad_enabled(False)
                valid_progress_bar = tqdm(range(1, len(valid_dataloader)), mininterval=5.)
                for j in valid_progress_bar:
                    try:
                        data_valid = next(tr_it_valid)
                    except StopIteration:
                        tr_it_valid = iter(valid_dataloader)
                        data_valid = next(tr_it_valid)
                    scene_valid = data_valid[0].to(device)
                    scene_valid = scene_valid.permute(0, 3, 2, 1)
                    his_traj_valid = data_valid[1].to(device)
                    his_traj_valid = his_traj_valid.permute(1, 0, 2)
                    targets_valid = data_valid[2].to(device)
                    _, valid_ade, pred, conf = forward(scene_valid.float(), his_traj_valid.float(),
                                                       targets_valid.float(), model, device, compute_loss=True)
                    valid_fde = utils._final_displacement_error(targets_valid, pred, conf, mode='best')
                    valid_ade = valid_ade.item()
                    valid_fde = valid_fde.item()
                    valid_ades.append(valid_ade)
                    valid_fdes.append(valid_fde)
                mean_ade_valid = np.mean(valid_ades)
                mean_fde_valid = np.mean(valid_fdes)
                torch.set_grad_enabled(True)
                train_writer.add_scalar('mean_ade_valid', mean_ade_valid, i)
                train_writer.add_scalar('mean_fde_valid', mean_fde_valid, i)
                print('eval phase','epoch: %5d' % epoch_i, 'i: %5d' % i, 'ade(avg): %10.5f' % mean_ade_valid,
                      'fde(avg): %10.5f' % mean_fde_valid, end=' | \n')

        torch.save(model.state_dict(), f'../model/{model_name}_final.pth')
        torch.save(optimizer.state_dict(), f'../model/{model_name}_optimizer_final.pth')
        results = pd.DataFrame({
            'iterations': iterations,
            'metrics_g (avg)': metrics,
            'elapsed_time (mins)': times,
        })
        results.to_csv(f'../log/train_metrics_{model_name}.csv', index=False)