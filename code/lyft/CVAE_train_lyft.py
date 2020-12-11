import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
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

import l5kit
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from torchvision.models.resnet import resnet50, resnet18, resnet34, resnet101
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
from pathlib import Path
import cvae
import utils
import matplotlib.pyplot as plt


import os
import time
import random

import warnings
warnings.filterwarnings("ignore")
from IPython.display import display
from tqdm import tqdm_notebook
import gc, psutil

print(l5kit.__version__)

cfg = {
    'format_version': 4,
    'data_path': '/home/chris/predict_code/Prediction/datasets/lyft-motion-prediction-autonomous-vehicles',
    'model_params': {
        'model_architecture': 'resnet34',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        'model_name': "CVAE_resnet34",
        'lr': 15e-4,
        'weight_path': '',
        'train': True,
        'predict': False,
    },
    'raster_params': {
        'raster_size': [400, 400],
        'pixel_size': [0.3, 0.3],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5,
    },
    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 12,
        'shuffle': True,
        'num_workers': 4,
    },
    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 128,
        'shuffle': False,
        'num_workers': 4,
    },
    'train_params': {
        'multi': 'best',
        'steps': 200000,
        'update_steps': 1000,
        'metrics_steps': 1000,
        'checkpoint_steps': 1000,
    }
}


# 主训练
def forward(data, model, device, criterion=cvae.loss_cvae, compute_loss=True):
    image = data["image"].to(device)
    history_traj = data["history_positions"].flip(1)
    history_yaw = data["history_yaws"].flip(1)
    history_availabilities = torch.unsqueeze(data["history_availabilities"].flip(1), 2)
    history = torch.cat([history_traj, history_yaw], 2)
    history = torch.cat([history, history_availabilities], 2).to(device)
    # print(history_traj.shape, history_yaw.shape, history_availabilities.shape)
    history = torch.cat([history_traj, history_yaw], 2)
    history = torch.cat([history, history_availabilities], 2).to(device)

    target_availabilities = data["target_availabilities"].to(device)
    targets = data["target_positions"].to(device)
    # history_traj: bz * seq_len *
    # print(image.shape, history.shape)
    preds, confidences, _, z_mean, z_var = model(image, history.permute(1, 0, 2))
    # skip compute loss if we are doing prediction
    loss, loss_pred = criterion(targets, preds, confidences, target_availabilities, z_mean,
                                z_var) if compute_loss else 0
    return loss, loss_pred, preds, confidences


if __name__ == '__main__':
    # 加载数据集，准备device
    DIR_INPUT = cfg["data_path"]
    os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
    dm = LocalDataManager()
    rasterizer = build_rasterizer(cfg, dm)
    train_cfg = cfg["train_data_loader"]
    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open(cached=False)  # to prevent run out of memory
    train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"],
                                  batch_size=train_cfg["batch_size"], num_workers=train_cfg["num_workers"])
    print(train_dataset, len(train_dataset))

    train_writer = SummaryWriter('../log/CVAE', comment='CVAE')

    # 建立模型
    model = cvae.CVAE(cfg, traj_dim=256, cont_dim=256, latent_dim=128, mode_dim=3, v_dim=4)
    weight_path = cfg["model_params"]["weight_path"]
    if weight_path:
        model.load_state_dict(torch.load(weight_path))
        print(weight_path, 'loaded')
    # print(model)
    model.cuda()
    learning_rate = cfg["model_params"]["lr"]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.96)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device {device}')

    # 开始训练
    if cfg["model_params"]["train"]:
        tr_it = iter(train_dataloader)
        n_steps = cfg["train_params"]["steps"]
        progress_bar = tqdm(range(1, 1 + n_steps), mininterval=5.)
        iterations = []
        losses = []
        losses_pre = []
        losses_ade = []
        losses_fde = []
        metrics = []
        metrics_pre = []
        metrics_ade = []
        metrics_fde = []
        times = []
        model_name = cfg["model_params"]["model_name"]
        multi_mode = cfg["train_params"]["multi"]
        update_steps = cfg['train_params']['update_steps']
        metrics_steps = cfg['train_params']['metrics_steps']
        checkpoint_steps = cfg['train_params']['checkpoint_steps']
        t_start = time.time()
        torch.set_grad_enabled(True)
        print('start train, mode is ', multi_mode)

        for i in progress_bar:
            try:
                data = next(tr_it)
            except StopIteration:
                tr_it = iter(train_dataloader)
                data = next(tr_it)
            model.train()  # somehow we need this is ever batch or it perform very bad (not sure why)
            loss, loss_pre, pred, conf = forward(data, model, device)

            # Backward pass
            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_v = loss.item()
            loss_pre_v = loss_pre.item()
            losses.append(loss_v)
            losses_pre.append(loss_pre_v)
            train_writer.add_scalar('loss', loss_v, i)
            train_writer.add_scalar('loss_pre', loss_pre_v, i)

            if i % metrics_steps == 0:
                # 求其他尺度指标
                loss_ade = utils._average_displacement_error(data["target_positions"].to(device),
                                                             pred, conf, data["target_availabilities"].to(device),
                                                             mode=multi_mode)
                loss_fde = utils._final_displacement_error(data["target_positions"].to(device),
                                                           pred, conf, data["target_availabilities"].to(device),
                                                           mode=multi_mode)
                losses_ade.append(loss_ade.item())
                losses_fde.append(loss_fde.item())
                train_writer.add_scalar('loss_ade', loss_ade, i)
                train_writer.add_scalar('loss_fde', loss_fde, i)

            if i % update_steps == 0:
                mean_losses = np.mean(losses)
                mean_losses_pre = np.mean(losses_pre)
                losses = []
                losses_pre = []
                train_writer.add_scalar('mean_loss', mean_losses, i)
                train_writer.add_scalar('mean_loss_pre', mean_losses_pre, i)
                mean_losses_ade = np.mean(losses_ade)
                mean_losses_fde = np.mean(losses_fde)
                losses_ade = []
                losses_fde = []
                train_writer.add_scalar('loss_ade', mean_losses_ade, i)
                train_writer.add_scalar('loss_fde', mean_losses_fde, i)
                timespent = (time.time() - t_start) / 60
                lr = optimizer.param_groups[0]['lr']
                train_writer.add_scalar('lr', lr, i)
                print('i: %5d' % i,
                      'loss(avg): %10.5f' % mean_losses, 'loss_pre(avg): %10.5f' % mean_losses_pre,
                      'loss_ade(avg): %10.5f' % mean_losses_ade, 'loss_fde(avg): %10.5f' % mean_losses_fde,
                      'lr(curr): %f' % lr, ' %.2fmins' % timespent, end=' | \n')
                if i % checkpoint_steps == 0:
                    torch.save(model.state_dict(), f'../model/{model_name}.pth')
                    torch.save(model, f'../model/{model_name}.pkl')
                    torch.save(optimizer.state_dict(), f'../model/{model_name}_optimizer.pth')
                iterations.append(i)
                metrics.append(mean_losses)
                metrics_pre.append(mean_losses_pre)
                metrics_ade.append(mean_losses_ade)
                metrics_fde.append(mean_losses_fde)
                times.append(timespent)

        torch.save(model.state_dict(), f'../model/{model_name}_final.pth')
        torch.save(model, f'../model/{model_name}_final.pkl')
        torch.save(optimizer.state_dict(), f'../model/{model_name}_optimizer_final.pth')
        results = pd.DataFrame({
            'iterations': iterations,
            'metrics (avg)': metrics,
            'metrics_pre (avg)': metrics_pre,
            'metrics_ade (avg)': metrics_ade,
            'metrics_fde (avg)': metrics_fde,
            'elapsed_time (mins)': times,
        })
        results.to_csv(f'train_metrics_{model_name}_{n_steps}.csv', index=False)