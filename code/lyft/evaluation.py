import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
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
import matplotlib.pyplot as plt
import tqdm


import os
import time
import random

import warnings
warnings.filterwarnings("ignore")
from IPython.display import display
from tqdm import tqdm_notebook
import gc, psutil

print('l5kit version', l5kit.__version__)

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
        'lr': 2e-5,
        'weight_path': '../model/CVAE_resnet34.pth',
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
    'val_data_loader': {
        'key': "scenes/validate.zarr",
        'batch_size': 10,
        'shuffle': False,
        'num_workers': 4
    }
}

DIR_INPUT = cfg["data_path"]
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT


def visualize_trajectory(dataset, index, title="target_positions movement with draw_trajectory"):
    data = dataset[index]
    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)
    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
    draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, radius=1, yaws=data["target_yaws"])

    plt.title(title)
    plt.imshow(im[::-1])
    plt.show()


if __name__ == '__main__':
    dm = LocalDataManager(None)
    rasterizer = build_rasterizer(cfg, dm)
    num_frames_to_chop = 100
    eval_cfg = cfg["val_data_loader"]
    eval_base_path = create_chopped_dataset(dm.require(eval_cfg["key"]),
                                            cfg["raster_params"]["filter_agents_threshold"],
                                            num_frames_to_chop, cfg["model_params"]["future_num_frames"],
                                            MIN_FUTURE_STEPS)
    eval_zarr_path = str(Path(eval_base_path) / Path(dm.require(eval_cfg["key"])).name)
    eval_mask_path = str(Path(eval_base_path) / "mask.npz")
    eval_gt_path = str(Path(eval_base_path) / "gt.csv")

    eval_zarr = ChunkedDataset(eval_zarr_path).open()
    eval_mask = np.load(eval_mask_path)["arr_0"]
    # ===== INIT DATASET AND LOAD MASK
    eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer, agents_mask=eval_mask)
    eval_dataloader = DataLoader(eval_dataset, shuffle=eval_cfg["shuffle"], batch_size=eval_cfg["batch_size"],
                                 num_workers=eval_cfg["num_workers"])
    print(eval_dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_path = cfg["model_params"]["weight_path"]
    model = torch.load(weight_path)
    model.cuda()
    torch.set_grad_enabled(False)

    if cfg["model_params"]["predict"]:

        model.eval()
        torch.set_grad_enabled(False)

        # store information for evaluation
        future_coords_offsets_pd = []
        timestamps = []
        confidences_list = []
        agent_ids = []

        progress_bar = tqdm(eval_dataloader)

        for data in progress_bar:
            inputs = data["image"].to(device)
            target_availabilities = data["target_availabilities"].to(device)
            targets = data["target_positions"].to(device)
            preds, confidences = model(inputs, device)

            # fix for the new environment
            preds = preds.cpu().numpy()
            world_from_agents = data["world_from_agent"].numpy()
            centroids = data["centroid"].numpy()
            coords_offset = []

            # convert into world coordinates and compute offsets
            for idx in range(len(preds)):
                for mode in range(3):
                    preds[idx, mode, :, :] = transform_points(preds[idx, mode, :, :], world_from_agents[idx]) - \
                                             centroids[idx][:2]

            future_coords_offsets_pd.append(preds.copy())
            confidences_list.append(confidences.cpu().numpy().copy())
            timestamps.append(data["timestamp"].numpy().copy())
            agent_ids.append(data["track_id"].numpy().copy())
        pred_path = 'submission1.csv'
        write_pred_csv(pred_path,
                       timestamps=np.concatenate(timestamps),
                       track_ids=np.concatenate(agent_ids),
                       coords=np.concatenate(future_coords_offsets_pd),
                       confs=np.concatenate(confidences_list)
                       )
        metrics = compute_metrics_csv(eval_gt_path, pred_path, [neg_multi_log_likelihood, time_displace])
        for metric_name, metric_mean in metrics.items():
            print(metric_name, metric_mean)

        gt_rows = {}
        for row in read_gt_csv(eval_gt_path):
            gt_rows[row["track_id"] + row["timestamp"]] = row["coord"]

        eval_ego_dataset = EgoDataset(cfg, eval_dataset.dataset, rasterizer)

        for frame_number in range(99, len(eval_zarr.frames),
                                  100):  # start from last frame of scene_0 and increase by 100
            agent_indices = eval_dataset.get_frame_indices(frame_number)
            if not len(agent_indices):
                continue

            # get AV point-of-view frame
            data_ego = eval_ego_dataset[frame_number]
            im_ego = rasterizer.to_rgb(data_ego["image"].transpose(1, 2, 0))
            center = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

            predicted_positions = []
            target_positions = []

            for v_index in agent_indices:
                data_agent = eval_dataset[v_index]

                out_net = model(torch.from_numpy(data_agent["image"]).unsqueeze(0).to(device))
                out_pos = out_net[0].reshape(-1, 2).detach().cpu().numpy()
                # store absolute world coordinates
                predicted_positions.append(out_pos + data_agent["centroid"][:2])
                # retrieve target positions from the GT and store as absolute coordinates
                track_id, timestamp = data_agent["track_id"], data_agent["timestamp"]
                target_positions.append(gt_rows[str(track_id) + str(timestamp)] + data_agent["centroid"][:2])

            # convert coordinates to AV point-of-view so we can draw them
            predicted_positions = transform_points(np.concatenate(predicted_positions), data_ego["world_to_image"])
            target_positions = transform_points(np.concatenate(target_positions), data_ego["world_to_image"])

            yaws = np.zeros((len(predicted_positions), 1))
            draw_trajectory(im_ego, predicted_positions, yaws, PREDICTED_POINTS_COLOR)
            draw_trajectory(im_ego, target_positions, yaws, TARGET_POINTS_COLOR)

            plt.imshow(im_ego[::-1])
            plt.show()