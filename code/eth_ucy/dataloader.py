import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset


class EthSceneDataset(Dataset):
    def __init__(self, data_dir, obs_len=8, pred_len=12):
        super(EthSceneDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        self.dataset = []
        for path in all_files:
            data = np.load(path, allow_pickle=True)
            self.dataset.append(data)
            del data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        scene = self.dataset[item]['scene']
        obs_traj = self.dataset[item]['obs_traj'][:, 0:2]
        pred_traj = self.dataset[item]['pred_traj']
        obs_traj_pixel = self.dataset[item]['obs_traj_pixel'][:, 0:2]
        pred_traj_pixel = self.dataset[item]['pred_traj_pixel']
        out = [scene, obs_traj, pred_traj, obs_traj_pixel, pred_traj_pixel]

        return tuple(out)
