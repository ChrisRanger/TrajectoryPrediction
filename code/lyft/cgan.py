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


class generator(nn.Module):
    def __init__(self, cfg: Dict, traj_dim=256, cont_dim=256, mode=3, v_dim=4):
        super(generator, self).__init__()
        self.Traj_Encoder = basicModel.Trajectory_Encoder(h_dim=traj_dim, v_dim=v_dim)
        self.Cont_Encoder = basicModel.Context_Encoder(cfg, cont_dim=cont_dim)

        self.future_len = cfg["model_params"]["future_num_frames"]
        self.num_modes = mode
        self.num_preds = 2 * self.future_len * self.num_modes
        self.gen = nn.Sequential(
            nn.BatchNorm1d(traj_dim + cont_dim),
            nn.Linear(in_features=traj_dim + cont_dim, out_features=traj_dim + cont_dim),
            nn.Linear(in_features=traj_dim + cont_dim, out_features=self.num_preds + self.num_modes)
        )

    def forward(self, image, traj):
        traj_vector = self.Traj_Encoder(traj).squeeze(0)
        cont_vector = self.Cont_Encoder(image)
        x = torch.cat([traj_vector, cont_vector], 1)
        x = self.gen(x)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences, cont_vector


class discriminator(nn.Module):
    def __init__(self, cdf:Dict, h_dim=64, cont_dim=256):
        super(discriminator, self).__init__()

        self.encoder = basicModel.Trajectory_Encoder(h_dim=h_dim, v_dim=2)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=h_dim+cont_dim, out_features=256),
            nn.Linear(in_features=256, out_features=1),
            nn.LeakyReLU()
        )

    def forward(self, pred: Tensor, condition: Tensor):
        latent = self.encoder(pred).squeeze(0)
        classify = torch.cat([latent, condition], dim=1)
        score = self.classifier(classify)
        return score
