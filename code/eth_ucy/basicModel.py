from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch import nn, optim
from torchvision.models.resnet import resnet50, resnet18, resnet34, resnet101
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
import matplotlib.pyplot as plt


import os
import time
import random

import warnings
warnings.filterwarnings("ignore")
from IPython.display import display
from tqdm import tqdm_notebook


# 轨迹向量编码，输出轨迹潜向量
class Trajectory_Encoder(nn.Module):
    def __init__(self, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0, v_dim=2):

        super(Trajectory_Encoder, self).__init__()
        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.v_dim = v_dim

        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        self.spatial_embedding = nn.Linear(v_dim, embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, obs_traj):
        # obs_traj: history_len * batch * input_size
        batch = obs_traj.size(1)
        # 历史轨迹重构为(obs_len * batch, v_dim)并送入embedding层(全连接)，成为
        # print(obs_traj.shape)
        obs_traj_embedding = self.spatial_embedding(obs_traj.contiguous().view(-1, self.v_dim))
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.embedding_dim
        )
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        # obs_traj: batch * h_dim
        return final_h


# 语义地图嵌入，输出condition
class Context_Encoder(nn.Module):
    def __init__(self, cnn_model='resnet34', channels=3, cont_dim=256):
        super().__init__()
        architecture = cnn_model
        backbone = eval(architecture)(pretrained=True, progress=True)
        self.backbone = backbone
        num_in_channels = channels
        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )
        if architecture == "resnet50":
            backbone_out_features = 2048
        else:
            backbone_out_features = 512

        # 全连接层
        self.head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features, out_features=cont_dim),
        )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x


# 单轨迹解码生成
class Singel_Decoder(nn.Module):
    def __init__(
        self, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        dropout=0.0, activation='relu', batch_norm=True
    ):
        super(Singel_Decoder, self).__init__()
        self.pred_len = 12
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim

        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, c_context, h_traj, last_pos):

        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        latent_state = (h_traj, c_context)

        for _ in range(self.seq_len):
            # 2维坐标embed
            decoder_input = self.spatial_embedding(last_pos)
            # 调整维度为batchx1xembedding_dim
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            # 以一个时刻长度输入LSTM
            output, state_tuple = self.decoder(decoder_input, latent_state)
            cur_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = cur_pos + last_pos
            last_pos = curr_pos
            # 时序上逐个扩张拼接起来， seq_len x batchsize x 2
            pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        # 交换维度为batchsize x seq_len x  2
        return pred_traj_fake_rel.permute(1, 0, 2)
