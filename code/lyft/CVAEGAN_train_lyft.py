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
import utils
import cgan
import cvae
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
    'data_path': '/home/yx/WSY/Prediction/datasets/lyft-motion-prediction-autonomous-vehicles',
    'model_params': {
        'model_architecture': 'resnet34',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        'model_name_g': "CVAEGAN_resnet34_G",
        'model_name_d': "CVAEGAN_resnet34_D",
        'lr_g': 15e-4,
        'lr_d': 15e-4,
        'weight_path_g': '',
        'weight_path_d': '',
        'train': True,
        'predict': False,
    },
    'raster_params': {
        'raster_size': [224, 224],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5,
    },
    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 16,
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
        'steps': 400000,
        'update_steps': 1000,
        'metrics_steps': 100,
        'checkpoint_steps': 2000,
    }
}


# 生成器训练
def forward_g(data, model_g, model_d, device, optimizer, scheduler, criterion=utils.g_loss, omega=10.0):
    image = data["image"].to(device)
    history_traj = data["history_positions"].flip(1)
    history_yaw = data["history_yaws"].flip(1)
    history_availabilities = torch.unsqueeze(data["history_availabilities"].flip(1), 2)
    history = torch.cat([history_traj, history_yaw], 2)
    history = torch.cat([history, history_availabilities], 2).to(device)
    target_availabilities = data["target_availabilities"].to(device)
    targets = data["target_positions"].to(device)

    model_g.train()
    preds, confidences, condition, z_mean, z_var = model_g(image, history.permute(1, 0, 2))
    traj_fake = multi2single(preds, targets, target_availabilities, confidences, mode='best')
    score_fake = model_d(traj_fake.permute(1, 0, 2), condition)
    # 判别loss + vae_loss
    g_loss = criterion(score_fake)
    cvae_output = cvae.loss_cvae(targets, preds, confidences, target_availabilities, z_mean, z_var)
    cvae_loss = cvae_output[0]
    loss = g_loss * omega + cvae_loss
    scheduler.step()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, cvae_loss, preds, confidences


# 判别器训练
def forward_d(data, model_g, model_d, device, optimizer, scheduler, criterion=utils.d_loss):
    image = data["image"].to(device)
    history_traj = data["history_positions"].flip(1)
    history_yaw = data["history_yaws"].flip(1)
    history_availabilities = torch.unsqueeze(data["history_availabilities"].flip(1), 2)
    history = torch.cat([history_traj, history_yaw], 2)
    history = torch.cat([history, history_availabilities], 2).to(device)
    target_availabilities = data["target_availabilities"].to(device)
    targets = data["target_positions"].to(device)

    model_d.train()
    preds, confidences, condition, _, _ = model_g(image, history.permute(1, 0, 2))
    traj_fake = multi2single(preds, targets, target_availabilities, confidences, mode='best')
    score_fake = model_d(traj_fake.permute(1, 0, 2), condition)
    score_real = model_d(targets.permute(1, 0, 2), condition)
    loss = criterion(score_real, score_fake)
    scheduler.step()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


# 多模态轨迹挑选函数，支持多种模态捕捉方式
def multi2single(traj_multi, ground_truth, avails, confidence, mode='best'):
    # 三种模式： best最接近gt那条; mean求期望; most取概率最大那条，其他方式待补充
    if mode == 'most':
        index = confidence.argmax(dim=1)
        selected_traj = gt
        for i in range(0, ground_truth.shape[0]):
            selected_traj[i] = traj_multi[i][index[i]]
        selected_traj = selected_traj.squeeze(1)
        return selected_traj
    elif mode == 'mean':
        traj_multi[:][0] = traj_multi[:][0] * confidence[0]
        traj_multi[:][1] = traj_multi[:][1] * confidence[1]
        traj_multi[:][2] = traj_multi[:][2] * confidence[2]
        traj_multi = torch.sum(traj_multi, dim=1)
        return traj_multi / 3.0
    elif mode == 'best':
        gt = torch.unsqueeze(ground_truth, 1)
        avails = avails[:, None, :, None]
        # 求L2距离平方
        error = torch.sum(((gt - traj_multi) * avails) ** 2, dim=3)
        # 时序轴上求平均
        error = torch.mean(error, dim=2)
        # 模态轴上求最小值
        index = error.argmin(dim=1)
        selected_traj = torch.tensor(gt)
        for i in range(0, ground_truth.shape[0]):
            selected_traj[i] = traj_multi[i][index[i]]
        selected_traj = selected_traj.squeeze(1)
        return selected_traj
    else:
        print('error, non-exist mode for select traj from multi!')


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

    train_writer = SummaryWriter('../log/CVAEGAN', comment='CVAEGAN')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 建立模型
    generator = cvae.CVAE(cfg, traj_dim=256, cont_dim=256, mode_dim=3, v_dim=4)
    discriminator = cgan.discriminator(cfg, h_dim=256, cont_dim=256)
    weight_path_g = cfg["model_params"]["weight_path_g"]
    if weight_path_g:
        generator.load_state_dict(torch.load(weight_path))
    weight_path_d = cfg["model_params"]["weight_path_d"]
    if weight_path_d:
        discriminator.load_state_dict(torch.load(weight_path))
    generator.cuda()
    discriminator.cuda()
    g_loss_fn = utils.g_loss
    d_loss_fn = utils.d_loss

    learning_rate_g = cfg["model_params"]["lr_g"]
    learning_rate_d = cfg["model_params"]["lr_d"]
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate_d)

    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=1000, gamma=0.96)
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=1000, gamma=0.96)
    print(f'device {device}')
    torch.backends.cudnn.benchmark = True

    # 开始训练
    if cfg["model_params"]["train"]:
        tr_it = iter(train_dataloader)
        n_steps = cfg["train_params"]["steps"]
        progress_bar = tqdm(range(1, 1 + n_steps), mininterval=5.)
        losses_g = []
        losses_d = []
        losses_vae = []
        losses_ade = []
        losses_fde = []
        iterations = []
        metrics_g = []
        metrics_d = []
        metrics_vae = []
        metrics_ade = []
        metrics_fde = []
        times = []
        model_name_g = cfg["model_params"]["model_name_g"]
        model_name_d = cfg["model_params"]["model_name_d"]
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

            # 判别器训练
            loss_d = forward_d(data, generator, discriminator, device,
                               optimizer_d, scheduler_d, criterion=d_loss_fn)

            loss_d = loss_d.item()
            losses_d.append(loss_d)
            train_writer.add_scalar('loss_d', loss_d, i)

            # 生成器训练
            loss_g, loss_vae, preds, confidences = forward_g(data, generator, discriminator,
                                                             device, optimizer_g, scheduler_g,
                                                             criterion=g_loss_fn, omega=10.0)
            loss_g = loss_g.item()
            losses_g.append(loss_g)
            train_writer.add_scalar('loss_g', loss_g, i)
            losses_vae.append(loss_vae.item())
            train_writer.add_scalar('loss_vae', loss_vae, i)

            if i % metrics_steps == 0:
                # 求其他尺度指标
                loss_ade = utils._average_displacement_error(data["target_positions"].to(device),
                                                             preds, confidences,
                                                             data["target_availabilities"].to(device),
                                                             mode=multi_mode)
                loss_fde = utils._final_displacement_error(data["target_positions"].to(device),
                                                           preds, confidences,
                                                           data["target_availabilities"].to(device),
                                                           mode=multi_mode)
                losses_ade.append(loss_ade.item())
                losses_fde.append(loss_fde.item())
                train_writer.add_scalar('loss_ade', loss_ade, i)
                train_writer.add_scalar('loss_fde', loss_fde, i)

            if i % update_steps == 0:
                mean_losses_g = np.mean(losses_g)
                mean_losses_d = np.mean(losses_d)
                train_writer.add_scalar('mean_loss_g', mean_losses_g, i)
                train_writer.add_scalar('mean_loss_d', mean_losses_d, i)
                mean_losses_ade = np.mean(losses_ade)
                mean_losses_fde = np.mean(losses_fde)
                mean_losses_vae = np.mean(losses_vae)
                losses_g = []
                losses_d = []
                losses_vae = []
                losses_ade = []
                losses_fde = []
                train_writer.add_scalar('loss_vae', mean_losses_vae, i)
                train_writer.add_scalar('loss_ade', mean_losses_ade, i)
                train_writer.add_scalar('loss_fde', mean_losses_fde, i)
                timespent = (time.time() - t_start) / 60
                curr_lr_g = optimizer_g.param_groups[0]['lr']
                curr_lr_d = optimizer_d.param_groups[0]['lr']
                print('i: %5d' % i,
                      'loss_g(avg): %10.5f' % mean_losses_g, 'loss_vae(avg): %10.5f' % mean_losses_vae,
                      'loss_d(avg): %10.5f' % mean_losses_d, 'loss_ade(avg): %10.5f' % mean_losses_ade,
                      'loss_fde(avg): %10.5f' % mean_losses_fde, 'lr(g): %f' % curr_lr_g, 'lr(d): %f' % curr_lr_d,
                      ' %.2fmins' % timespent, end=' | \n')
                if i % checkpoint_steps == 0:
                    torch.save(generator, f'../model/{model_name_g}.pkl')
                    torch.save(generator.state_dict(), f'../model/{model_name_g}.pth')
                    torch.save(optimizer_g.state_dict(), f'../model/{model_name_g}_optimizer.pth')
                    torch.save(discriminator, f'../model/{model_name_d}.pkl')
                    torch.save(discriminator.state_dict(), f'../model/{model_name_d}.pth')
                    torch.save(optimizer_d.state_dict(), f'../model/{model_name_d}_optimizer.pth')
                iterations.append(i)
                metrics_g.append(mean_losses_g)
                metrics_d.append(mean_losses_d)
                metrics_ade.append(mean_losses_ade)
                metrics_fde.append(mean_losses_fde)
                metrics_vae.append(mean_losses_vae)
                times.append(timespent)

        torch.save(generator.state_dict(), f'../model/{model_name_g}_final.pth')
        torch.save(generator, f'../model/{model_name_g}_final.pkl')
        torch.save(optimizer_g.state_dict(), f'../model/{model_name_g}_optimizer_final.pth')
        torch.save(discriminator, f'../model/{model_name_d}.pkl')
        torch.save(discriminator.state_dict(), f'../model/{model_name_d}.pth')
        torch.save(optimizer_d.state_dict(), f'../model/{model_name_d}_optimizer.pth')
        results = pd.DataFrame({
            'iterations': iterations,
            'metrics_g (avg)': metrics_g,
            'metrics_d (avg)': metrics_d,
            'metrics_ade (avg)': metrics_ade,
            'metrics_fde (avg)': metrics_fde,
            'metrics_vae (avg)': metrics_vae,
            'elapsed_time (mins)': times,
        })
        results.to_csv(f'../log/train_metrics_cvaegan_{n_steps}.csv', index=False)