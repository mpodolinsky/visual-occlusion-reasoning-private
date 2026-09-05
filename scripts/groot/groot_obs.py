"""Bridge LIBERO observations <-> GR00T LIBERO_PANDA policy.

Ported from ``16-LIBERO-X-GR00T-ZeroShot/sim/groot_obs.py``, which mirrors
NVIDIA's own ``gr00t/eval/sim/LIBERO/libero_env.py`` (n1.7-release):

  - images: agentview / eye-in-hand, rotated 180 deg (``[::-1, ::-1]``), passed
    at the raw env resolution (256); the GR00T processor resizes. This is the
    same 180 deg correction ``eval_pi05_libero.rotate_camera_image`` applies.
  - state: [eef_pos(3), quat2axisangle(eef_quat)(3), gripper_qpos(2)] = 8,
    laid into the x/y/z/roll/pitch/yaw/gripper slots.
  - action: 7-dim delta-EEF. The gripper dim comes back in [0, 1] and is
    normalised to [-1, 1], binarised, then sign-inverted before ``env.step``.

Set ``LIBEROX_NATIVE_CONVENTION=1`` to instead match a checkpoint fine-tuned on
``meituan/LIBERO-X`` (vertical-flip-only images + gripper passthrough in
{-1, +1}). Off by default -- the ``nvidia/GR00T-N1.7-LIBERO`` checkpoints use
the NVIDIA convention.
"""

from __future__ import annotations

import math
import os

import numpy as np

_LIBEROX_NATIVE = os.environ.get("LIBEROX_NATIVE_CONVENTION", "0") == "1"

STATE_ORDER = ("x", "y", "z", "roll", "pitch", "yaw")
ACTION_ORDER = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
LANG_KEY = "annotation.human.action.task_description"


def quat2axisangle(quat) -> np.ndarray:
    """robosuite quaternion (x, y, z, w) -> axis-angle exponential coords."""
    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return q[:3] * 2.0 * math.acos(float(q[3])) / den


def build_flat_obs(obs: dict, prompt: str) -> dict:
    """LIBERO obs dict -> flat Gr00tSimPolicyWrapper observation (B=1, T=1)."""
    sl = (
        (slice(None, None, -1),)
        if _LIBEROX_NATIVE
        else (slice(None, None, -1), slice(None, None, -1))
    )
    img = np.ascontiguousarray(obs["agentview_image"][sl]).astype(np.uint8)
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][sl]).astype(np.uint8)

    xyz = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    rpy = quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32)
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)

    return {
        "video.image": img[None, None],           # (1,1,H,W,3) uint8
        "video.wrist_image": wrist[None, None],
        "state.x": xyz[0:1][None, None],           # (1,1,1) float32
        "state.y": xyz[1:2][None, None],
        "state.z": xyz[2:3][None, None],
        "state.roll": rpy[0:1][None, None],
        "state.pitch": rpy[1:2][None, None],
        "state.yaw": rpy[2:3][None, None],
        "state.gripper": gripper[None, None],      # (1,1,2) float32
        LANG_KEY: [str(prompt)],                   # (B,) list[str]
        "task": [str(prompt)],                     # alias; ignored unless the config asks for it
    }


def decode_action_step(raw_action) -> list[float]:
    """One (7,) GR00T action row -> robosuite OSC_POSE action for env.step."""
    a = np.asarray(raw_action, dtype=np.float32).copy()
    if _LIBEROX_NATIVE:
        a[-1] = np.sign(a[-1])       # model already emits raw robosuite gripper in {-1, +1}
    else:
        a[-1] = 2.0 * a[-1] - 1.0    # normalize_gripper_action: [0, 1] -> [-1, 1]
        a[-1] = np.sign(a[-1])       # binarise
        a[-1] = -a[-1]               # invert_gripper_action
    return a.tolist()
