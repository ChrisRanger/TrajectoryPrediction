import numpy as np
from typing import List, Tuple, Dict
import torch
from torch import Tensor
import random
import math
from typing import List, Tuple
from torch.nn import functional as f

def pytorch_neg_multi_log_likelihood_batch(
    gt: Tensor, pred: Tensor, confidences: Tensor) -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    _assert_shapes(gt, pred, confidences)

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes

    # error (batch_size, num_modes, future_len)
    error = torch.sum((gt - pred) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return torch.mean(error)


def pytorch_neg_multi_log_likelihood_single(
    gt: Tensor, pred: Tensor, avails: Tensor
) -> Tensor:
    """

    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(time)x(2D coords)
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    # create confidence (bs)x(mode=1)
    batch_size, future_len, num_coords = pred.shape
    confidences = pred.new_ones((batch_size, 1))
    return pytorch_neg_multi_log_likelihood_batch(gt, pred.unsqueeze(1), confidences, avails)


def _assert_shapes(ground_truth: Tensor, pred: Tensor, confidences: Tensor) -> None:
    assert len(pred.shape) == 4, f"expected 4D (BSxMxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape
    assert ground_truth.shape == (batch_size, future_len, num_coords), \
        f"expected 2D (Time x Coords) array for gt, got {ground_truth.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    s = torch.sum(confidences, dim=1)
    assert torch.allclose(s, torch.ones_like(s)), "confidences should sum to 1"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(ground_truth).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"


def _average_displacement_error(
    ground_truth: Tensor, pred: Tensor, confidences: Tensor, mode: str
) -> Tensor:

    _assert_shapes(ground_truth, pred, confidences)
    gt = torch.unsqueeze(ground_truth, 1)  # add modes

    error = torch.sum((gt - pred) ** 2, dim=3)
    error = error ** 0.5  # calculate root of error (= L2 norm)
    error = torch.mean(error, dim=2)  # average over timesteps
    if mode == "best":
        error = torch.min(error, dim=1)  # 仅选最接近gt的那个模态轨迹
    elif mode == "mean":
        error = torch.mean(error, dim=1)  # 所有模态轨迹求平均
    else:
        raise ValueError(f"mode: {mode} not valid")

    # batchsize 求均值
    return torch.mean(error.values, dim=0)

def average_displacement_error(
    ground_truth: Tensor, pred: Tensor) -> Tensor:
    gt = ground_truth
    error = torch.sum((gt - pred) ** 2, dim=2)
    error = error ** 0.5  # calculate root of error (= L2 norm)
    error = torch.mean(error, dim=1)  # average over timesteps
    # batchsize 求均值
    return torch.mean(error.values, dim=0)


def _final_displacement_error(
    ground_truth: Tensor, pred: Tensor, confidences: Tensor, mode: str
) -> Tensor:

    _assert_shapes(ground_truth, pred, confidences)
    gt = torch.unsqueeze(ground_truth, 1)  # add modes

    error = torch.sum((gt - pred) ** 2, dim=3)  # reduce coords and use availability
    error = error ** 0.5  # calculate root of error (= L2 norm)
    error = error[:, :, -1]  # use last timestep
    if mode == "best":
        error = torch.min(error, dim=1)  # use best hypothesis
    elif mode == "mean":
        error = torch.mean(error, dim=1)  # average over hypotheses
    else:
        raise ValueError(f"mode: {mode} not valid")

    # batchsize 求均值
    return torch.mean(error.values, dim=0)

def final_displacement_error(
    ground_truth: Tensor, pred: Tensor) -> Tensor:

    gt = ground_truth
    error = torch.sum((gt - pred) ** 2, dim=2)  # reduce coords and use availability
    error = error ** 0.5  # calculate root of error (= L2 norm)
    error = error[:, -1]  # use last timestep
    # batchsize 求均值
    return torch.mean(error.values, dim=0)


def bce_loss(input, target):
    # 二分类loss,手动实现sigmod激活函数
    # input: DG(z)
    # return:
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def g_loss(scores_fake):
    # scores_fake: DG(z)
    # return: E[log(1-DG(z))
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)


def d_loss(scores_real, scores_fake):
    # scores_real: D(x), scores_fake: DG(z)
    # return: E[log(D(x))] + E[log(1-DG(z))
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake

# 多模态轨迹挑选函数，支持多种模态捕捉方式
def multi2single(traj_multi, ground_truth, confidence, mode='best'):
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
        # 求L2距离平方
        error = torch.sum((gt - traj_multi) ** 2, dim=3)
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