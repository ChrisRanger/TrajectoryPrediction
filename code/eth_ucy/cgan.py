from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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


class generator(nn.Module):
    def __init__(self, cnn_model='resnet34', channels=3, traj_dim=256, cont_dim=256, mode=3, v_dim=4):
        super(generator, self).__init__()
        self.Traj_Encoder = basicModel.Trajectory_Encoder(h_dim=traj_dim, v_dim=v_dim)
        self.Cont_Encoder = basicModel.Context_Encoder(cnn_model='resnet34', channels=3, cont_dim=cont_dim)

        self.future_len = 12
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
    def __init__(self, h_dim=64, cont_dim=256):
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
