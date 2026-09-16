"""Vendored copy of infinite-hands/infinite-hands's hardware/yam_kinematics.py: forward kinematics
of one YAM arm, joint angles -> grasp-point position in the arm base frame. Keep in sync with that
file by hand -- this fork can't import the main repo.

Backend-agnostic on purpose (`xp` is numpy or jax.numpy) so the same function scores stored
predictions offline and differentiates inside the openpi training loss.
"""

from __future__ import annotations

import math

import numpy as np

from openpi.training.misc.ih_hardware_constants import LEFT_ARM_DIMS, RIGHT_ARM_DIMS

# --- kinematic chain, transcribed from i2rt yam.urdf joint1..joint6 (origin xyz, origin rpy, axis) ---
# The gripper mounts on link6 with an identity transform (linear_4310.yml last_joint_mount.yam ==
# joint6's origin), so the grasp site is expressed directly in the joint6 child frame.
_JOINTS = (
    ((-2.49777947129965e-07, -9.17402363407408e-13, 0.0679999999995413), (0.0, 1.5708, 3.1415963267949), (-1.0, 0.0, 0.0)),
    ((-0.0455, -0.0339, -0.02), (-1.7953e-16, 5.55112e-17, -3.14159), (3.58252e-15, 1.0, -3.2007e-16)),
    ((4.08431e-07, -0.0688, 0.264), (-2.99502e-14, -1.6015e-14, -3.14159), (4.54273e-15, 1.0, 1.04275e-14)),
    ((-0.0600003, -0.0688, -0.244999), (2.95536e-14, 8.52374e-14, -2.10585e-14), (0.0, 1.0, 0.0)),
    ((-0.0405003, 0.0338995, -0.0739989), (2.73264e-14, -4.32956e-16, -8.32667e-17), (1.0, 0.0, 0.0)),
    ((0.0404996, 2.39858e-07, -0.0356), (8.83309e-16, -5.88588e-16, 7.69861e-14), (1.46014e-15, 2.40808e-14, -1.0)),
)
GRASP_SITE_OFFSET = (0.0, 0.0, -0.1347)  # linear_4310.xml grasp_site: the point between the finger tips
ARM_JOINT_COUNT = 6
LEFT_JOINT_DIMS = LEFT_ARM_DIMS[:ARM_JOINT_COUNT]
RIGHT_JOINT_DIMS = RIGHT_ARM_DIMS[:ARM_JOINT_COUNT]


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx  # URDF convention: R = Rz(yaw) Ry(pitch) Rx(roll)


ORIGIN_ROTATIONS = np.stack([_rpy_matrix(*rpy) for _, rpy, _ in _JOINTS])            # (6, 3, 3)
ORIGIN_TRANSLATIONS = np.array([xyz for xyz, _, _ in _JOINTS])                          # (6, 3)
JOINT_AXES = np.array([np.asarray(axis) / np.linalg.norm(axis) for _, _, axis in _JOINTS])  # (6, 3)


def _axis_rotation(axis, angle, xp):
    """Rodrigues rotation about a unit axis; `angle` may carry any leading batch shape."""
    ax, ay, az = axis[0], axis[1], axis[2]
    zero = xp.zeros_like(angle)
    skew = xp.stack([
        xp.stack([zero, -az + zero, ay + zero], axis=-1),
        xp.stack([az + zero, zero, -ax + zero], axis=-1),
        xp.stack([-ay + zero, ax + zero, zero], axis=-1),
    ], axis=-2)
    sin = xp.sin(angle)[..., None, None]
    cos = xp.cos(angle)[..., None, None]
    eye = xp.eye(3, dtype=angle.dtype)
    return eye + sin * skew + (1.0 - cos) * (skew @ skew)


def fk_grasp_point(joints, xp=np):
    """Arm joint angles `[..., 6]` (radians) -> grasp-point position `[..., 3]` in the base frame."""
    joints = xp.asarray(joints)
    batch_shape = joints.shape[:-1]
    rotation = xp.broadcast_to(xp.eye(3, dtype=joints.dtype), (*batch_shape, 3, 3))
    position = xp.zeros((*batch_shape, 3), dtype=joints.dtype)
    for index in range(ARM_JOINT_COUNT):
        origin_rotation = xp.asarray(ORIGIN_ROTATIONS[index], dtype=joints.dtype)
        origin_translation = xp.asarray(ORIGIN_TRANSLATIONS[index], dtype=joints.dtype)
        position = position + (rotation @ origin_translation[..., None])[..., 0]
        rotation = rotation @ origin_rotation @ _axis_rotation(
            xp.asarray(JOINT_AXES[index], dtype=joints.dtype), joints[..., index], xp)
    offset = xp.asarray(GRASP_SITE_OFFSET, dtype=joints.dtype)
    return position + (rotation @ offset[..., None])[..., 0]


def grasp_points(actions14, xp=np):
    """Absolute 14-dim bimanual joints `[..., 14]` -> (left, right) grasp points, each `[..., 3]`."""
    actions14 = xp.asarray(actions14)
    left = fk_grasp_point(actions14[..., LEFT_JOINT_DIMS[0]:LEFT_JOINT_DIMS[-1] + 1], xp)
    right = fk_grasp_point(actions14[..., RIGHT_JOINT_DIMS[0]:RIGHT_JOINT_DIMS[-1] + 1], xp)
    return left, right
