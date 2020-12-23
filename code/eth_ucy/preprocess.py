import pickle
import glob
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import math
import random
from shutil import copy2

cfg = {
    'params': {
        'history_num_frames': 8,
        'delta_time': 0.25,
        'future_num_frames': 12,
    },
    'eth': {
        'data_path': '/datasets/ETH/seq_eth/',
        'scene_map': '/datasets/ETH/seq_eth/raw_map.png',
        'width': 640,
        'heigth': 480,
        },
    'hotel': {
        'data_path': '/datasets/ETH/seq_hotel/',
        'scene_map': '/datasets/ETH/seq_hotel/map.png',
        'width': 720,
        'heigth': 576,
        },
    'univ': {
        'data_path': '/hd/yx/WSY/UCY/students03/',
        'scene_map': '/hd/yx/WSY/UCY/students03/map.png',
        'width': 720,
        'heigth': 576,
        },
    'zara1': {
        'data_path': '/hd/yx/WSY/UCY/zara01/',
        'scene_map': '/hd/yx/WSY/UCY/zara01/map.png',
        'width': 720,
        'heigth': 576,
        },
    'zara2': {
        'data_path': '/hd/yx/WSY/UCY/zara02/',
        'scene_map': '/hd/yx/WSY/UCY/zara02/map.png',
        'width': 720,
        'heigth': 576,
        },
}


if __name__ == "__main__":
  params = cfg['params']
  dataset = cfg['univ']
  width = dataset['width']
  heigth = dataset['heigth']
  raw_traj = np.genfromtxt(dataset['data_path'] + 'obsmat.txt')
  raw_pixel = np.genfromtxt(dataset['data_path'] + 'obsmat_px.txt')

  print(raw_traj.shape, raw_pixel.shape)
  raw_traj = np.delete(raw_traj, [3, 6], axis=1)
  raw_pixel = np.delete(raw_pixel, [3, 6], axis=1)
  print(raw_traj.shape, raw_pixel.shape)
  raw_video = cv2.VideoCapture(os.path.join(dataset['data_path'], 'video.avi'))
  # raw_video.set(cv2.CAP_PROP_POS_FRAMES, 1000)
  # flag, image = raw_video.read()

  h_matrix = np.genfromtxt(os.path.join(dataset['data_path'], 'H.txt'))
  obs_len = params['history_num_frames']
  pred_len = params['future_num_frames']
  seq_len = obs_len + pred_len
  print(obs_len, pred_len, seq_len)

  frames = np.unique(raw_traj[:, 0]).tolist()
  frame_data = []
  frame_data_pixel = []
  for frame in frames:
    frame_data.append(raw_traj[frame == raw_traj[:, 0], :])  # 从data中按frame_id重排数据到frame_data
    frame_data_pixel.append(raw_pixel[frame == raw_pixel[:, 0], :])
  num_sequences = int(math.ceil(len(frames) - seq_len + 1))  # 总的帧数中去掉seq_len为总切片数
  i = 0
  for idx in range(0, num_sequences + 1):  # 遍历本数据集，抽取每帧所有行人数据
    curr_seq_data = np.concatenate(frame_data[idx:idx + seq_len], axis=0)  # 抽取从当前帧算起向后共seq_len帧数据
    curr_seq_data_pixel = np.concatenate(frame_data_pixel[idx:idx + seq_len], axis=0)
    curr_frame = frame_data[idx+obs_len][0][0]
    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])  # 抽取当前帧中所有行人id并去重
    # 按行人遍历seq_len帧的frame数据，抽取每位行人的坐标
    for _, ped_id in enumerate(peds_in_curr_seq):
      curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]  # 抽取同一行人id的轨迹到curr_ped_seq
      curr_ped_seq_pixel = curr_seq_data_pixel[curr_seq_data_pixel[:, 1] == ped_id, :]
      curr_ped_seq = np.around(curr_ped_seq, decimals=4)
      curr_ped_seq_pixel = np.around(curr_ped_seq_pixel, decimals=4)
      pad_front = frames.index(curr_ped_seq[0, 0]) - idx  # 获取当前行人序列最前frame_id
      pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1  # 获取当前行人序列最后frame_id
      if pad_end - pad_front != seq_len:  # 当前seq中帧数不够seq_len则跳过该行人
        continue
      curr_ped_seq = curr_ped_seq[:, 2:]  # 剔除前两列(frame_id和ped_id)，seq_len*4的tensor，每行为x,y,vx,vy
      curr_ped_seq_pixel = curr_ped_seq_pixel[:, 2:]
      curr_all = frame_data[idx+obs_len]
      curr_others = []
      for enity in curr_all:
        if enity[1] != ped_id:
          curr_others.append([enity[2], enity[3]])
      curr_all_pixel = frame_data_pixel[idx + obs_len]
      curr_others_pixel = []
      for enity in curr_all_pixel:
          if enity[1] != ped_id:
              curr_others_pixel.append([enity[2], enity[3]])

      # 渲染
      img = cv2.imread(dataset['scene_map'])
      # print('neignbor_num', len(curr_others))
      # ego
      x = int(curr_ped_seq_pixel[obs_len-1][1])
      y = int(curr_ped_seq_pixel[obs_len-1][0])

      cv2.circle(img, center=(x, y), radius=5, color=(0, 255, 0), thickness=-1)
      # others
      for neignbor in curr_others_pixel:
        x = int(neignbor[1])
        y = int(neignbor[0])
        # if x<width and y<heigth:
        cv2.circle(img, center=(x, y), radius=5, color=(0, 0, 255), thickness=-1)
      # raw_video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
      # flag, image = raw_video.read()
      # cv2.imshow('', image)
      # cv2.waitKey(0)
      # cv2.imshow('', img)
      # cv2.waitKey(0)

      # 转为相对与当前位置的轨迹点
      curr_pos = curr_ped_seq[obs_len-1, :]
      curr_ped_seq[:, 0] = curr_ped_seq[:, 0]
      curr_ped_seq[:, 1] = curr_ped_seq[:, 1]
      curr_pos_pixel = curr_ped_seq_pixel[obs_len - 1, :]
      curr_ped_seq_pixel[:, 0] = curr_ped_seq_pixel[:, 0]
      curr_ped_seq_pixel[:, 1] = curr_ped_seq_pixel[:, 1]

      data = {
        "frame_id": curr_frame,
        "ped_id": ped_id,
        "curr_pos": curr_pos,
        "obs_traj": curr_ped_seq[0:obs_len, :],
        "pred_traj": curr_ped_seq[obs_len:, :2],
        "curr_others": curr_others,
        "curr_pos_pixel": curr_pos_pixel,
        "obs_traj_pixel": curr_ped_seq_pixel[0:obs_len, :],
        "pred_traj_pixel": curr_ped_seq_pixel[obs_len:, :2],
        "curr_others_pixel": curr_others_pixel,
        "scene": img
      }
      curr_path = dataset['data_path'] + '/processed/' + f'{i}'
      np.savez(curr_path, **data)
      i += 1

  # 划分数据集7:2:1
  all_data = os.listdir(dataset['data_path'] + '/processed/')
  num_all_data = len(all_data)
  index_list = list(range(num_all_data))
  random.shuffle(index_list)
  trainDir = dataset['data_path'] + '/train/'
  if not os.path.exists(trainDir):
    os.mkdir(trainDir)

  validDir = dataset['data_path'] + '/valid/'
  if not os.path.exists(validDir):
    os.mkdir(validDir)

  testDir = dataset['data_path'] + '/test/'
  if not os.path.exists(testDir):
    os.mkdir(testDir)

  num = 0
  for i in index_list:
    fileName = os.path.join(dataset['data_path'] + '/processed', all_data[i])
    if num < num_all_data * 0.7:
      copy2(fileName, trainDir)
    elif num >= num_all_data * 0.7 and num < num_all_data * 0.8:
      copy2(fileName, validDir)
    else:
      copy2(fileName, testDir)
    num += 1

