"""Vendored copy of infinite-hands/infinite-hands's models/vla/pi/loss_math.py: the arithmetic of
the critical-window weighted loss, on numpy or jax.numpy arrays. Keep in sync with that file by
hand -- this fork can't import the main repo.

Everything here takes the normalized, delta-joint, zero-padded chunks openpi hands its model and
returns plain arrays.
"""

from __future__ import annotations

import numpy as np

from openpi.training.misc.ih_hardware_constants import GRIPPER_DIMS, JOINT_DIMS, STATE_DIM
from openpi.training.misc.ih_yam_kinematics import grasp_points

QUANTILE_EPSILON = 1e-6  # openpi's Normalize adds this to every denominator; matched exactly


def unnormalize_quantile(values, q01, q99, xp=np):
    """Invert openpi's quantile normalization on the first `len(q01)` dims; padding passes through."""
    values = xp.asarray(values)
    q01 = xp.asarray(q01, dtype=values.dtype)
    q99 = xp.asarray(q99, dtype=values.dtype)
    width = q01.shape[-1]
    raw = (values[..., :width] + 1.0) / 2.0 * (q99 - q01 + QUANTILE_EPSILON) + q01
    return xp.concatenate([raw, values[..., width:]], axis=-1)


def absolute_joints(actions, state, xp=np):
    """Delta-joint chunk `[..., H, 14]` + state `[..., 14]` -> absolute 14-dim commands (grippers as-is)."""
    actions = xp.asarray(actions)[..., :STATE_DIM]
    state = xp.asarray(state)[..., :STATE_DIM]
    joint_mask = xp.asarray(np.isin(np.arange(STATE_DIM), JOINT_DIMS), dtype=actions.dtype)
    return actions + state[..., None, :] * joint_mask


def critical_window(actions, velocity_ref, gain, sigma_frames, xp=np):
    """Per-timestep weights peaked where the recorded grippers move.

    Returns `(critical, weight)`: `critical` in [0, 1] is the smoothed gripper speed relative to
    `velocity_ref`, `weight = 1 + gain * critical` rescaled to mean 1 over the whole batch so the
    loss keeps the baseline's scale and an A/B at identical hyperparameters stays fair.
    """
    actions = xp.asarray(actions)
    grippers = actions[..., GRIPPER_DIMS]
    speed = xp.abs(grippers[..., 1:, :] - grippers[..., :-1, :]).sum(axis=-1)
    speed = xp.concatenate([speed, speed[..., -1:]], axis=-1)
    critical = xp.clip(speed / velocity_ref, 0.0, 1.0) @ _gaussian_kernel(actions.shape[-2], sigma_frames, xp)
    critical = xp.clip(critical, 0.0, 1.0)
    weight = 1.0 + gain * critical
    return critical, weight / xp.mean(weight)


def _gaussian_kernel(horizon, sigma_frames, xp):
    positions = xp.arange(horizon, dtype=xp.float32)
    kernel = xp.exp(-((positions[:, None] - positions[None, :]) ** 2) / (2.0 * sigma_frames**2))
    return kernel / kernel.sum(axis=0, keepdims=True)


def dim_weights(action_dim, gripper_gain, xp=np):
    """Per-dimension weights that boost the gripper dims and keep the mean weight at 1."""
    weights = np.ones(action_dim, dtype=np.float32)
    weights[GRIPPER_DIMS] = gripper_gain
    return xp.asarray(weights * action_dim / weights.sum())


def ee_error(predicted14, expected14, xy_weight, z_weight, xp=np):
    """Weighted squared grasp-point error (m^2) summed over both arms, `[..., 14]` -> `[...]`."""
    axis_weights = xp.asarray([xy_weight, xy_weight, z_weight], dtype=xp.asarray(predicted14).dtype)
    total = 0.0
    for predicted, expected in zip(grasp_points(predicted14, xp), grasp_points(expected14, xp)):
        total = total + (((predicted - expected) ** 2) * axis_weights).sum(axis=-1)
    return total


def transition_index(gripper, velocity_ref, xp=np):
    """Recorded transition per chunk `[..., H]` -> (index `[...]`, present mask `[...]`)."""
    speed = xp.abs(gripper[..., 1:] - gripper[..., :-1])
    return xp.argmax(speed, axis=-1), speed.max(axis=-1) > 0.5 * velocity_ref


def soft_transition_index(gripper, temperature, xp=np):
    """Differentiable transition time of a predicted gripper chunk `[..., H]` -> `[...]`."""
    speed = xp.abs(gripper[..., 1:] - gripper[..., :-1]) / temperature
    logits = speed - speed.max(axis=-1, keepdims=True)
    weights = xp.exp(logits) / xp.exp(logits).sum(axis=-1, keepdims=True)
    return (weights * xp.arange(speed.shape[-1], dtype=weights.dtype)).sum(axis=-1)


def timing_error(predicted, expected, velocity_ref, temperature, xp=np):
    """Per-chunk |predicted - recorded| gripper transition offset, in horizon fractions, over both grippers."""
    horizon = predicted.shape[-2]
    total = 0.0
    for dim in GRIPPER_DIMS:
        index, present = transition_index(expected[..., dim], velocity_ref, xp)
        soft = soft_transition_index(predicted[..., dim], temperature, xp)
        total = total + present * xp.abs(soft - index) / horizon
    return total
