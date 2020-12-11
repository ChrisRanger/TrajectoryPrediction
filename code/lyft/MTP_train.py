from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torchsummary import summary
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
        'model_name': "model_resnet34_output",
        'lr': 1e-7,
        'weight_path': '../model/model_resnet34_output_i.pth',
        'train': True,
        'predict': False,
    },
    'raster_params': {
        'raster_size': [300, 300],
        'pixel_size': [0.4, 0.4],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5,
    },
    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 4,
    },
    'valid_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 2,
        'shuffle': True,
        'num_workers': 4,
    },
    'train_params': {
        'steps': 400000,
        'update_steps': 1000,
        'checkpoint_steps': 1000,
    }
}


class LyftMultiModel(nn.Module):
    def __init__(self, cfg: Dict, num_modes=3):
        super().__init__()

        architecture = cfg["model_params"]["model_architecture"]
        backbone = eval(architecture)(pretrained=True, progress=True)
        self.backbone = backbone

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )

        # This is 512 for resnet18 and resnet34
        # And it is 2048 for the other resnets
        if architecture == "resnet50":
            backbone_out_features = 2048
        else:
            backbone_out_features = 512

        # X, Y coords for the future positions (output shape: batch_sizex50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len

        # You can add more layers here.
        self.head = nn.Sequential(
            # nn.Dropout(0.2),
	    # nn.BatchNorm1d(backbone_out_features),
            nn.Linear(in_features=backbone_out_features, out_features=4096),
            nn.Linear(in_features=4096, out_features=2048),
	    nn.LeakyReLU(),
        )

        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.logit = nn.Linear(2048, out_features=self.num_preds + num_modes)

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
        x = self.logit(x)

        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences


def forward(data, model, device, criterion=utils.pytorch_neg_multi_log_likelihood_batch, compute_loss=True):
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].to(device)
    targets = data["target_positions"].to(device)
    # Forward pass
    preds, confidences = model(inputs)
    # skip compute loss if we are doing prediction
    loss = criterion(targets, preds, confidences, target_availabilities) if compute_loss else 0
    return loss, preds, confidences


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
    print(train_dataset)

    rasterizer = build_rasterizer(cfg, dm)
    valid_cfg = cfg["valid_data_loader"]
    valid_zarr = ChunkedDataset(dm.require(valid_cfg["key"])).open(cached=False)  # to prevent run out of memory
    valid_dataset = AgentDataset(cfg, valid_zarr, rasterizer)
    valid_dataloader = DataLoader(valid_dataset, shuffle=valid_cfg["shuffle"], batch_size=valid_cfg["batch_size"], num_workers=valid_cfg["num_workers"])
    print(valid_dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    train_writer = SummaryWriter('../log/MTP', comment='MTP')

    # 建立模型
    model = LyftMultiModel(cfg)

    # load weight if there is a pretrained model
    weight_path = cfg["model_params"]["weight_path"]
    if weight_path:
        model.load_state_dict(torch.load(weight_path))
        print(weight_path, "loaded")

    model.to(device)
    learning_rate = cfg["model_params"]["lr"]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
    print(f'device {device}')
    # raster_size = ((cfg["model_params"]["history_num_frames"] + 1) * 2 + 3, cfg["raster_params"]["raster_size"][0], cfg["raster_params"]["raster_size"][1])
    # summary(model, input_size=raster_size)
    # print(model)
    torch.backends.cudnn.benchmark = True

    # 开始训练
    if cfg["model_params"]["train"]:
        tr_it = iter(train_dataloader)
        tr_it_valid = iter(valid_dataloader)
        n_steps = cfg["train_params"]["steps"]
        valid_steps = 1000
        progress_bar = tqdm(range(1, 1 + n_steps), mininterval=5.)
        losses = []
        iterations = []
        metrics = []
        times = []
        model_name = cfg["model_params"]["model_name"]
        update_steps = cfg['train_params']['update_steps']
        checkpoint_steps = cfg['train_params']['checkpoint_steps']
        t_start = time.time()
        torch.set_grad_enabled(True)

        for i in progress_bar:
            try:
                data = next(tr_it)
            except StopIteration:
                tr_it = iter(train_dataloader)
                data = next(tr_it)

            model.train()  # somehow we need this is ever batch or it perform very bad (not sure why)
            loss, _, _ = forward(data, model, device)
	    # Backward pass
            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            losses.append(loss)
            train_writer.add_scalar('loss', loss, i)

            if i % update_steps == 0:
                mean_losses = np.mean(losses)
                losses = []
                train_writer.add_scalar('mean_losses', mean_losses, i)
                timespent = (time.time() - t_start) / 60
                curr_lr = optimizer.param_groups[0]['lr']
                train_writer.add_scalar('lr', curr_lr, i)
                print('i: %5d' % i,
                      'loss(avg): %10.5f' % mean_losses, 'lr(curr): %f' % curr_lr,
                      ' %.2fmins' % timespent, end=' | \n')
                if i % checkpoint_steps == 0:
                    torch.save(model, f'../model/{model_name}_{i}.pkl')
                    torch.save(model.state_dict(), f'../model/{model_name}_{i}.pth')
                    torch.save(optimizer.state_dict(), f'../model/{model_name}_optimizer_{i}.pth')
                iterations.append(i)
                metrics.append(mean_losses)
                times.append(timespent)

            if i % update_steps == 0:
                model.eval()
                losses_valid = []
                torch.set_grad_enabled(False)
                valid_progress_bar = tqdm(range(1, 1 + valid_steps), mininterval=5.)
                for j in valid_progress_bar :
                    try:
                        data_valid = next(tr_it_valid)
                    except StopIteration:
                        tr_it_valid = iter(valid_dataloader)
                        data_valid = next(tr_it_valid)
                    loss_valid, _, _ = forward(data_valid, model, device, compute_loss=True)
                    loss_valid = loss_valid.item()
                    losses_valid.append(loss_valid)
                    # train_writer.add_scalar('loss_valid', loss_valid, j)
                mean_loss_valid = np.mean(losses_valid)
                torch.set_grad_enabled(True)
                train_writer.add_scalar('mean_loss_valid', mean_loss_valid, i)
                print('eval finished, loss_valid(avg): %10.5f' % mean_loss_valid, end=' | \n')

        torch.save(model.state_dict(), f'../model/{model_name}_final.pth')
        torch.save(model, f'../model/{model_name}_final.pkl')
        torch.save(optimizer.state_dict(), f'../model/{model_name}_optimizer_final.pth')
        results = pd.DataFrame({
            'iterations': iterations,
            'metrics_g (avg)': metrics,
            'elapsed_time (mins)': times,
        })
        results.to_csv(f'../log/train_metrics_cgan_{n_steps}.csv', index=False)
