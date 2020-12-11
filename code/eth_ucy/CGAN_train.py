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

from pathlib import Path
import cgan
import utils
import dataloader
import scripts
import matplotlib.pyplot as plt

import os
import time
import random
import logging
from logging import handlers

logging.info(torch.__version__)

cfg = {
    'data_path': '/datasets/ETH/seq_eth/',
    'model_params': {
        'model_cnn': "resnet34",
        'scene_weight': 640,
        'scene_height': 480,
        'history_num_frames': 8,
        'history_delta_time': 0.25,
        'future_num_frames': 12,
        'model_name': "CGAN",
        'lr_g': 1e-5,
        'lr_d': 1e-3,
        'checkpoint_path': '',
        'train': True,
        'predict': True,
    },
    'train_data_loader': {
        'batch_size': 6,
        'shuffle': True,
        'num_workers': 4,
    },
    'valid_data_loader': {
        'batch_size': 6,
        'shuffle': True,
        'num_workers': 4,
    },
    'test_data_loader': {
        'batch_size': 4,
        'shuffle': False,
        'num_workers': 4,
    },
    'train_params': {
        'epoch': 2,
        'checkpoint_steps': 100,
        'valid_steps': 1,
        'log_file_path': '../../log/cgan.log',
        'omega': 1.0,
        'epsilon': 1.0,
    }
}


# 生成器训练
def forward_g(scene, his_traj, targets, model_g, model_d, optimizer, scheduler, omega=1.0, epsilon=1.0):

    model_g.train()
    preds, conf, context = model_g(scene, his_traj)
    traj_fake = utils.multi2single(preds, targets, conf, mode='best')
    score_fake = model_d(traj_fake.permute(1, 0, 2), context)
    # 判别loss + nll_loss
    g_loss = utils.g_loss(score_fake)
    nll_loss = utils.pytorch_neg_multi_log_likelihood_batch(targets, preds, conf)
    min_l2_loss = utils._average_displacement_error(targets, preds, conf, mode='best')
    #     loss = g_loss + nll_loss * omega + l2_loss * epsilon
    loss = g_loss + nll_loss * omega + min_l2_loss * epsilon
    scheduler.step()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, nll_loss, min_l2_loss, preds, conf


# 判别器训练
def forward_d(scene, his_traj, targets, model_g, model_d, optimizer, scheduler):

    model_d.train()
    preds, confidences, context = model_g(scene, his_traj)
    traj_fake = utils.multi2single(preds, targets, confidences, mode='best')
    score_fake = model_d(traj_fake.permute(1, 0, 2), context)
    score_real = model_d(targets.permute(1, 0, 2), context)
    loss = utils.d_loss(score_real, score_fake)
    scheduler.step()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


if __name__ == '__main__':
    logfile = cfg['train_params']['log_file_path']
    logger = logging.getLogger(logfile)
    logger.setLevel(level=logging.INFO)
    sh = logging.StreamHandler()  # 往屏幕上输出
    th = handlers.TimedRotatingFileHandler(filename=logfile, when='D', encoding='utf-8')
    logger.addHandler(sh)  # 把对象加到logger里
    logger.addHandler(th)

    # 加载数据集，准备device
    DIR_INPUT = cfg["data_path"]

    train_cfg = cfg["train_data_loader"]
    train_dataset = dataloader.EthSceneDataset(os.path.join(cfg['data_path'], 'train'))
    train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"],
                                  batch_size=train_cfg["batch_size"], num_workers=train_cfg["num_workers"],
                                  drop_last=True)

    valid_cfg = cfg["valid_data_loader"]
    valid_dataset = dataloader.EthSceneDataset(os.path.join(cfg['data_path'], 'valid'))
    valid_dataloader = DataLoader(valid_dataset, shuffle=valid_cfg["shuffle"], batch_size=valid_cfg["batch_size"],
                                  num_workers=valid_cfg["num_workers"], drop_last=True)

    test_cfg = cfg["test_data_loader"]
    test_dataset = dataloader.EthSceneDataset(os.path.join(cfg['data_path'], 'test'))
    test_dataloader = DataLoader(test_dataset, shuffle=test_cfg["shuffle"], batch_size=test_cfg["batch_size"],
                                  num_workers=test_cfg["num_workers"], drop_last=True)

    h_matrix = np.genfromtxt(DIR_INPUT + 'H.txt')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_writer = SummaryWriter('../../log/eth/cgan', comment='cvae')

    # 建立模型
    generator = cgan.generator(cnn_model=cfg["model_params"]["model_cnn"],
                      channels=3, cont_dim=256)
    discriminator = cgan.discriminator(h_dim=256, cont_dim=256)
    # load weight if there is a pretrained model
    checkpoint_path = cfg["model_params"]["checkpoint_path"]
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['model_state_dict_g'])
        discriminator.load_state_dict(checkpoint['model_state_dict_d'])
        logger.info(checkpoint_path, "loaded")
    generator.cuda()
    discriminator.cuda()

    learning_rate_g = cfg["model_params"]["lr_g"]
    learning_rate_d = cfg["model_params"]["lr_d"]
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate_d)

    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=20000, gamma=1.0)
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=2000, gamma=0.8)
    logger.info(f'device {device}')
    torch.backends.cudnn.benchmark = True

    # train
    if cfg["model_params"]["train"]:
        tr_it = iter(train_dataloader)
        tr_it_valid = iter(valid_dataloader)
        epochs = cfg["train_params"]["epoch"]
        progress_bar = tqdm(range(1, len(train_dataloader)), mininterval=5.)
        losses_d = []
        losses_g = []
        losses_nll = []
        losses_ade = []
        model_name = cfg["model_params"]["model_name"]
        checkpoint_steps = cfg['train_params']['checkpoint_steps']
        valid_steps = cfg['train_params']['valid_steps']
        t_start = time.time()
        torch.set_grad_enabled(True)
        best_valid = [0., 0.]
        omega = cfg['train_params']['omega']
        epsilon = cfg['train_params']['epsilon']
        i = 0
        for epoch_i in range(epochs):
            for _ in progress_bar:
                try:
                    data = next(tr_it)
                except StopIteration:
                    tr_it = iter(train_dataloader)
                    data = next(tr_it)
                scene = data[0].to(device)
                scene = scene.permute(0, 3, 2, 1)
                his_traj = data[3].to(device)
                his_traj = his_traj.permute(1, 0, 2)
                targets = data[4].to(device)

                # 判别器训练
                loss_d = forward_d(scene.float(), his_traj.float(), targets.float(),
                                   generator, discriminator, optimizer_d, scheduler_d)
                loss_d = loss_d.item()
                losses_d.append(loss_d)
                train_writer.add_scalar('train/loss_d', loss_d, i)

                # 生成器训练
                loss_g, loss_nll, loss_ade, preds, confidences = forward_g(scene.float(), his_traj.float(),
                                                                           targets.float(), generator, discriminator,
                                                                           optimizer_g, scheduler_g,
                                                                           omega=omega, epsilon=epsilon)
                loss_g = loss_g.item()
                losses_g.append(loss_g)
                train_writer.add_scalar('train/loss_g', loss_g, i)
                loss_nll = loss_nll.item()
                losses_nll.append(loss_nll)
                train_writer.add_scalar('train_metrics/loss_nll', loss_nll, i)
                loss_ade = loss_ade.item()
                losses_ade.append(loss_ade)
                train_writer.add_scalar('train_metrics/loss_ade', loss_nll, i)

                i += 1

                if i % checkpoint_steps == 0:
                    mean_losses_d = np.mean(losses_d)
                    losses_d = []
                    train_writer.add_scalar('train_mean/mean_losses_d', mean_losses_d, i)
                    mean_losses_g = np.mean(losses_g)
                    losses_g = []
                    train_writer.add_scalar('train_mean/mean_losses_g', mean_losses_g, i)

                    timespent = (time.time() - t_start) / 60
                    curr_lr_g = optimizer_g.param_groups[0]['lr']
                    curr_lr_d = optimizer_d.param_groups[0]['lr']
                    train_writer.add_scalar('lr/lr_g', curr_lr_g, i)
                    train_writer.add_scalar('lr/lr_d', curr_lr_d, i)
                    logger.info('epoch: {}, i: {}, loss_g(avg): {},'
                                 'loss_d(avg): {}, time(min):{}'.format(epoch_i, i,
                                                                        mean_losses_g, mean_losses_d, timespent))

            # valid
            if epoch_i % valid_steps == 0:
                generator.eval()
                valid_ades = []
                valid_fdes = []
                valid_losses = []
                torch.set_grad_enabled(False)
                valid_progress_bar = tqdm(range(1, len(valid_dataloader)), mininterval=5.)
                for j in valid_progress_bar:
                    try:
                        data_valid = next(tr_it_valid)
                    except StopIteration:
                        tr_it_valid = iter(valid_dataloader)
                        data_valid = next(tr_it_valid)
                    scene_valid = data_valid[0].to(device)
                    scene_valid = scene_valid.permute(0, 3, 2, 1)
                    his_traj_valid = data_valid[3].to(device)
                    his_traj_valid = his_traj_valid.permute(1, 0, 2)
                    targets_valid = data_valid[4].to(device)
                    pred_pixel, conf, context = generator(scene_valid.float(), his_traj_valid.float())
                    traj_fake_valid = utils.multi2single(pred_pixel, targets_valid.float(), conf, mode='best')
                    score_fake = discriminator(traj_fake_valid.permute(1, 0, 2), context)
                    g_loss_valid = utils.g_loss(score_fake)
                    nll_loss_valid = utils.pytorch_neg_multi_log_likelihood_batch(targets_valid, pred_pixel, conf)
                    min_l2_loss_valid = utils._average_displacement_error(targets_valid, pred_pixel, conf, mode='best')
                    #     loss = g_loss + nll_loss * omega + l2_loss * epsilon
                    valid_loss = g_loss_valid + nll_loss_valid * omega + min_l2_loss_valid * epsilon
                    # camera frame to world frame(meter)
                    pred = torch.zeros_like(pred_pixel)
                    for batch_index in range(pred_pixel.shape[0]):
                        for modality in range(pred_pixel.shape[1]):
                            for pos_index in range(pred_pixel.shape[2]):
                                pred[batch_index][modality][pos_index] = torch.from_numpy(scripts.project(h_matrix,
                                        pred_pixel[batch_index][modality][pos_index].cpu()))
                    # calculate metrics in world frame
                    valid_ade = utils._average_displacement_error(data_valid[2].to(device), pred, conf, mode='best')
                    valid_fde = utils._final_displacement_error(data_valid[2].to(device), pred, conf, mode='best')
                    valid_loss = valid_loss.item()
                    valid_ade = valid_ade.item()
                    valid_fde = valid_fde.item()
                    valid_losses.append(valid_loss)
                    valid_ades.append(valid_ade)
                    valid_fdes.append(valid_fde)
                mean_loss_valid = np.mean(valid_losses)
                mean_ade_valid = np.mean(valid_ades)
                mean_fde_valid = np.mean(valid_fdes)
                # 仅在模型提升时更新checkpoint
                if mean_ade_valid<best_valid[0] and mean_fde_valid<best_valid[1]:
                    checkpoint = {"model_state_dict_g": generator.state_dict(),
                                  "optimizer_state_dict_g": optimizer_g.state_dict(),
                                  "model_state_dict_d": discriminator.state_dict(),
                                  "optimizer_state_dict_d": optimizer_d.state_dict(),
                                  "epoch": epoch_i}
                    path_checkpoint = os.path.join('../../model/', 'cgan_model.pt')
                    torch.save(checkpoint, path_checkpoint)
                best_valid[0] = mean_ade_valid
                best_valid[1] = mean_fde_valid

                torch.set_grad_enabled(True)
                train_writer.add_scalar('valid/mean_loss_valid', mean_loss_valid, i)
                train_writer.add_scalar('valid/mean_ade_valid', mean_ade_valid, i)
                train_writer.add_scalar('valid/mean_fde_valid', mean_fde_valid, i)
                logger.info('eval phase: epoch: {}, i: {}, loss(avg): {}, ade(avg): {}, fde(avg): {}'.format(epoch_i, i, mean_loss_valid,
                                                                                mean_ade_valid, mean_fde_valid))

    # test
    if cfg["model_params"]["predict"]:
        tr_it_test = iter(test_dataloader)
        progress_bar = tqdm(range(1, len(test_dataloader)), mininterval=5.)
        test_ades = []
        test_fdes = []
        times = []
        model_name = cfg["model_params"]["model_name"]
        t_start = time.time()
        generator.eval()
        torch.set_grad_enabled(False)
        logging.info('test phase')

        for _ in progress_bar:
            try:
                data_test = next(tr_it_test)
            except StopIteration:
                tr_it_test = iter(test_dataloader)
                data_test = next(tr_it_test)
            scene_test = data_test[0].to(device)
            scene_test = scene_test.permute(0, 3, 2, 1)
            his_traj_test = data_test[3].to(device)
            his_traj_test = his_traj_test.permute(1, 0, 2)
            targets_test = data_test[4].to(device)
            pred_pixel_test, conf_test, _ = generator(scene_test.float(), his_traj_test.float())
            # camera frame to world frame(meter)
            pred_test = torch.zeros_like(pred_pixel_test)
            for batch_index in range(pred_pixel_test.shape[0]):
                for modality in range(pred_pixel_test.shape[1]):
                    for pos_index in range(pred_pixel_test.shape[2]):
                        pred_test[batch_index][modality][pos_index] = torch.from_numpy(scripts.project(h_matrix,
                                    pred_pixel_test[batch_index][modality][pos_index].cpu()))
            # calculate metrics in world frame
            test_ade = utils._average_displacement_error(data_test[2].to(device), pred_test, conf_test, mode='best')
            test_fde = utils._final_displacement_error(data_test[2].to(device), pred_test, conf_test, mode='best')
            test_ade = test_ade.item()
            test_fde = test_fde.item()
            test_ades.append(test_ade)
            test_fdes.append(test_fde)
        mean_ade_test = np.mean(test_ades)
        mean_fde_test = np.mean(test_fdes)
        logger.info('test phase: ade(avg): {}, fde(avg): {}'.format(mean_ade_test, mean_fde_test))