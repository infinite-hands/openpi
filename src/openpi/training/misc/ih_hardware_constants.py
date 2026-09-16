"""Vendored subset of infinite-hands/infinite-hands's hardware/constants.py: just the 14-dim
bimanual YAM state/action layout the weighted loss needs. Keep in sync with that file by hand --
this fork can't import the main repo, so STATE_DIM/GRIPPER_DIMS/JOINT_DIMS/LEFT_ARM_DIMS/
RIGHT_ARM_DIMS are copied here rather than shared.
"""

# --- 14-dim bimanual layout: [L j0..5, L grip, R j0..5, R grip] (one schema; keep the block whole) ---
STATE_DIM = 14                                                       # observation.state size
GRIPPER_DIMS = [6, 13]                                               # the two gripper dims
JOINT_DIMS = [d for d in range(STATE_DIM) if d not in GRIPPER_DIMS]  # the 12 joint dims
LEFT_ARM_DIMS = [0, 1, 2, 3, 4, 5, 6]                                # left arm: joints 0-5 then gripper 6
RIGHT_ARM_DIMS = [7, 8, 9, 10, 11, 12, 13]                           # right arm: joints 7-12 then gripper 13
