from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
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
import basicModel
import matplotlib.pyplot as plt


import os
import time
import random

import warnings
warnings.filterwarnings("ignore")
from IPython.display import display
from tqdm import tqdm_notebook
import gc, psutil


class CVAE(nn.Module):
    def __init__(self, cfg: Dict, traj_dim=256, cont_dim=256, latent_dim=128, mode_dim=3, v_dim=4):
        super(CVAE, self).__init__()
        self.Traj_Encoder = basicModel.Trajectory_Encoder(h_dim=traj_dim, v_dim=v_dim)
        self.Cont_Encoder = basicModel.Context_Encoder(cfg, cont_dim=cont_dim)
        # 回归均值
        self.encoder_mean = nn.Sequential(
            nn.Linear(in_features=traj_dim+cont_dim, out_features=256),
            nn.Linear(in_features=256, out_features=latent_dim),
            # nn.BatchNorm1d(latent_dim)
        )
        # 回归方差
        self.encoder_var = nn.Sequential(
            nn.Linear(in_features=traj_dim+cont_dim, out_features=256),
            nn.Linear(in_features=256, out_features=latent_dim),
            # nn.BatchNorm1d(latent_dim)
        )

        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len
        self.num_preds = num_targets * mode_dim
        self.num_modes = mode_dim
        self.decoder_net = nn.Sequential(
            nn.BatchNorm1d(latent_dim + cont_dim),
            nn.Linear(in_features=latent_dim + cont_dim, out_features=2048),
            nn.Linear(in_features=2048, out_features=self.num_preds+mode_dim),
        )

    def encoder(self, image, traj):
        # image->condition, traj->input
        input = self.Traj_Encoder(traj).squeeze(0)
        condition = self.Cont_Encoder(image)
        combination = torch.cat([input, condition], 1)
        combination = F.elu(combination)
        z_mean = self.encoder_mean(combination)
        z_var = self.encoder_var(combination)
        return z_mean, z_var, condition

    def reparameterize(self, mean, var):
        # 求标准差
        std = torch.exp(0.5 * var)
        # 标准差范围内采样,加入噪声
        eps = torch.randn_like(std)
        return mean + eps * std

    def decoder(self, latent, condition):
        inputs = torch.cat([latent, condition], 1)
        x = self.decoder_net(inputs)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences

    def forward(self, image, traj):
        z_mean, z_var, context = self.encoder(image, traj)
        z_sample = self.reparameterize(z_mean, z_var)
        future_traj, conf = self.decoder(z_sample, context)
        return future_traj, conf, context, z_mean, z_var


def loss_cvae(gt_traj, future_traj, conf, aviable, mean, var):
    pred_loss = utils.pytorch_neg_multi_log_likelihood_batch(gt_traj, future_traj, conf, aviable)
    KLD_loss = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())
    return pred_loss+KLD_loss, pred_loss




