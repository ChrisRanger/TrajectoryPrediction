import numpy as np
from typing import List, Tuple, Dict
import torch
from torch import Tensor
import random
import math
from typing import List, Tuple
from torch.nn import functional as f
from l5kit.rasterization import semantic_rasterizer


def _assert_shapes(ground_truth: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor) -> None:
    assert len(pred.shape) == 4, f"expected 4D (BSxMxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert ground_truth.shape == (batch_size, future_len, num_coords), \
        f"expected 2D (Time x Coords) array for gt, got {ground_truth.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    s = torch.sum(confidences, dim=1)
    assert torch.allclose(s, torch.ones_like(s)), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(ground_truth).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"


def _average_displacement_error(
    ground_truth: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor, mode: str
) -> Tensor:

    _assert_shapes(ground_truth, pred, confidences, avails)
    gt = torch.unsqueeze(ground_truth, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    error = torch.sum(((gt - pred) * avails) ** 2, dim=3)
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


def _final_displacement_error(
    ground_truth: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor, mode: str
) -> Tensor:

    _assert_shapes(ground_truth, pred, confidences, avails)
    gt = torch.unsqueeze(ground_truth, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    error = torch.sum(((gt - pred) * avails) ** 2, dim=3)  # reduce coords and use availability
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


def pytorch_neg_multi_log_likelihood_batch(
    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
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
    _assert_shapes(gt, pred, confidences, avails)

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

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


def c_loss(cfg: Dict, dataset: semantic_rasterizer, pred: Tensor, condition: Tensor,
           curr_pos: Tensor, ego_pos: Tensor, mode='best'):
    '''
    求解预测的轨迹在当前BEV语义地图中的DAC(满足可行驶区域的轨迹点占所有轨迹点的比率)
    待测试
    '''
    assert len(pred.shape) == 4, f"expected 4D (BSxMxTxC) array for pred, got {pred.shape}"
    assert len(condition.shape) == 4, f"expected 4D (BSxchannelxheightxwidth) array for pred, got {condition.shape}"

    image = dataset.rasterizer.to_rgb(condition)

    pixel_size = cfg["raster_params"]['pixel_size']
    # 预测轨迹相对坐标转换到绝对坐标
    pred_pos = torch.add(pred[:, :, :, 0], curr_pos[0])
    pred_pos = torch.add(pred_pos[:, :, :, 1], curr_pos[1])

    # 白色为道路外, 红色为路口禁行,均不满足可行驶区域
    forbidden_color = ([255, 255, 255], [255, 0, 0])

    # 计算当前av所在的栅格位置作为参照
    ego_center = [cfg["raster_params"]['ego_center'][0]*cfg["raster_params"]['raster_size'][0],
                  cfg["raster_params"]['ego_center'][1]*cfg["raster_params"]['raster_size'][1]]

    dac_bs = []
    for i in pred_pos:
        for j in i:
            dac_mode = []
            for points in j:
                outside = 0
                x = (points[0] - ego_pos[0]) / pixel_size + ego_center[0]
                y = (points[1] - ego_pos[1]) / pixel_size + ego_center[1]
                if image[x][y] == forbidden_color[0] or image[x][y] == forbidden_color[1]:
                    outside += 1
                dac_mode.append(1.0 - outside/cfg["model_params"]['future_num_frames'])
            if mode == 'best':
                dac_bs.append(np.max(dac_mode))
            elif mode == 'mean':
                dac_bs.append(np.mean(dac_mode))
            else:
                print('error: non-exist DAC-mode in multi')
    return np.mean(dac_bs)


class MTPLoss:
    """ Computes the loss for the MTP model. """
    def __init__(self,
                 num_modes: int,
                 regression_loss_weight: float = 1.,
                 angle_threshold_degrees: float = 5.):
        """
        Inits MTP loss.
        :param num_modes: How many modes are being predicted for each agent.
        :param regression_loss_weight: Coefficient applied to the regression loss to
            balance classification and regression performance.
        :param angle_threshold_degrees: Minimum angle needed between a predicted trajectory
            and the ground to consider it a match.
        """
        self.num_modes = num_modes
        self.num_location_coordinates_predicted = 2  # We predict x, y coordinates at each timestep.
        self.regression_loss_weight = regression_loss_weight
        self.angle_threshold = angle_threshold_degrees

    def _get_trajectory_and_modes(self,
                                  model_prediction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits the predictions from the model into mode probabilities and trajectory.
        :param model_prediction: Tensor of shape [batch_size, n_timesteps * n_modes * 2 + n_modes].
        :return: Tuple of tensors. First item is the trajectories of shape [batch_size, n_modes, n_timesteps, 2].
            Second item are the mode probabilities of shape [batch_size, num_modes].
        """
        mode_probabilities = model_prediction[:, -self.num_modes:].clone()

        desired_shape = (model_prediction.shape[0], self.num_modes, -1, self.num_location_coordinates_predicted)
        trajectories_no_modes = model_prediction[:, :-self.num_modes].clone().reshape(desired_shape)

        return trajectories_no_modes, mode_probabilities

    @staticmethod
    def _angle_between(ref_traj: torch.Tensor,
                       traj_to_compare: torch.Tensor) -> float:
        """
        Computes the angle between the last points of the two trajectories.
        The resulting angle is in degrees and is an angle in the [0; 180) interval.
        :param ref_traj: Tensor of shape [n_timesteps, 2].
        :param traj_to_compare: Tensor of shape [n_timesteps, 2].
        :return: Angle between the trajectories.
        """
        EPSILON = 1e-5
        if (ref_traj.ndim != 2 or traj_to_compare.ndim != 2 or
                ref_traj.shape[1] != 2 or traj_to_compare.shape[1] != 2):
            raise ValueError('Both tensors should have shapes (-1, 2).')
        if torch.isnan(traj_to_compare[-1]).any() or torch.isnan(ref_traj[-1]).any():
            return 180. - EPSILON
        traj_norms_product = float(torch.norm(ref_traj[-1]) * torch.norm(traj_to_compare[-1]))
        # If either of the vectors described in the docstring has norm 0, return 0 as the angle.
        if math.isclose(traj_norms_product, 0):
            return 0.
        # We apply the max and min operations below to ensure there is no value
        # returned for cos_angle that is greater than 1 or less than -1.
        # This should never be the case, but the check is in place for cases where
        # we might encounter numerical instability.
        dot_product = float(ref_traj[-1].dot(traj_to_compare[-1]))
        angle = math.degrees(math.acos(max(min(dot_product / traj_norms_product, 1), -1)))
        if angle >= 180:
            return angle - EPSILON

        return angle

    @staticmethod
    def _compute_ave_l2_norms(tensor: torch.Tensor) -> float:
        """
        Compute the average of l2 norms of each row in the tensor.
        :param tensor: Shape [1, n_timesteps, 2].
        :return: Average l2 norm. Float.
        """
        l2_norms = torch.norm(tensor, p=2, dim=2)
        avg_distance = torch.mean(l2_norms)
        return avg_distance.item()

    def _compute_angles_from_ground_truth(self, target: torch.Tensor,
                                          trajectories: torch.Tensor) -> List[Tuple[float, int]]:
        """
        Compute angle between the target trajectory (ground truth) and the predicted trajectories.
        :param target: Shape [1, n_timesteps, 2].
        :param trajectories: Shape [n_modes, n_timesteps, 2].
        :return: List of angle, index tuples.
        """
        angles_from_ground_truth = []
        for mode, mode_trajectory in enumerate(trajectories):
            # For each mode, we compute the angle between the last point of the predicted trajectory for that
            # mode and the last point of the ground truth trajectory.
            angle = self._angle_between(target[0], mode_trajectory)

            angles_from_ground_truth.append((angle, mode))
        return angles_from_ground_truth

    def _compute_best_mode(self,
                           angles_from_ground_truth: List[Tuple[float, int]],
                           target: torch.Tensor, trajectories: torch.Tensor) -> int:
        """
        Finds the index of the best mode given the angles from the ground truth.
        :param angles_from_ground_truth: List of (angle, mode index) tuples.
        :param target: Shape [1, n_timesteps, 2]
        :param trajectories: Shape [n_modes, n_timesteps, 2]
        :return: Integer index of best mode.
        """

        # We first sort the modes based on the angle to the ground truth (ascending order), and keep track of
        # the index corresponding to the biggest angle that is still smaller than a threshold value.
        angles_from_ground_truth = sorted(angles_from_ground_truth)
        max_angle_below_thresh_idx = -1
        for angle_idx, (angle, mode) in enumerate(angles_from_ground_truth):
            if angle <= self.angle_threshold:
                max_angle_below_thresh_idx = angle_idx
            else:
                break

        # We choose the best mode at random IF there are no modes with an angle less than the threshold.
        if max_angle_below_thresh_idx == -1:
            best_mode = random.randint(0, self.num_modes - 1)

        # We choose the best mode to be the one that provides the lowest ave of l2 norms between the
        # predicted trajectory and the ground truth, taking into account only the modes with an angle
        # less than the threshold IF there is at least one mode with an angle less than the threshold.
        else:
            # Out of the selected modes above, we choose the final best mode as that which returns the
            # smallest ave of l2 norms between the predicted and ground truth trajectories.
            distances_from_ground_truth = []
            for angle, mode in angles_from_ground_truth[:max_angle_below_thresh_idx + 1]:
                norm = self._compute_ave_l2_norms(target - trajectories[mode, :, :])

                distances_from_ground_truth.append((norm, mode))

            distances_from_ground_truth = sorted(distances_from_ground_truth)
            best_mode = distances_from_ground_truth[0][1]

        return best_mode

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the MTP loss on a batch.
        The predictions are of shape [batch_size, n_ouput_neurons of last linear layer]
        and the targets are of shape [batch_size, 1, n_timesteps, 2]
        :param predictions: Model predictions for batch.
        :param targets: Targets for batch.
        :return: zero-dim tensor representing the loss on the batch.
        """

        batch_losses = torch.Tensor().requires_grad_(True).to(predictions.device)
        trajectories, modes = self._get_trajectory_and_modes(predictions)

        for batch_idx in range(predictions.shape[0]):
            angles = self._compute_angles_from_ground_truth(target=targets[batch_idx],
                                                            trajectories=trajectories[batch_idx])

            best_mode = self._compute_best_mode(angles,
                                                target=targets[batch_idx],
                                                trajectories=trajectories[batch_idx])

            best_mode_trajectory = trajectories[batch_idx, best_mode, :].unsqueeze(0)

            regression_loss = f.smooth_l1_loss(best_mode_trajectory, targets[batch_idx])

            mode_probabilities = modes[batch_idx].unsqueeze(0)
            best_mode_target = torch.tensor([best_mode], device=predictions.device)
            classification_loss = f.cross_entropy(mode_probabilities, best_mode_target)

            loss = classification_loss + self.regression_loss_weight * regression_loss

            batch_losses = torch.cat((batch_losses, loss.unsqueeze(0)), 0)

        avg_loss = torch.mean(batch_losses)
        return avg_loss
