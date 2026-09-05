"""GR00T-N1.7-LIBERO policy behind an openpi-style websocket.

Runs in the Isaac-GR00T `uv` venv (``submodules/Isaac-GR00T/.venv``). Speaks the
same msgpack-numpy protocol as openpi's WebsocketPolicyServer so the libero_10
collection side (``scripts/groot/collect.py``) can drive it with
``openpi_client.websocket_client_policy.WebsocketClientPolicy`` unchanged.

Ported verbatim from ``16-LIBERO-X-GR00T-ZeroShot/server/serve_groot_ws.py``
(default ``--model-path`` repointed at this repo's checkpoint dir) -- see
``VENDOR.md``.

Wire contract
-------------
On connect: server sends a packed metadata dict.
Per request: client sends a packed flat observation dict (the
`Gr00tSimPolicyWrapper` format, batch=1, time=1):
    video.image                              (1,1,H,W,3) uint8   (180deg-rotated agentview)
    video.wrist_image                        (1,1,H,W,3) uint8
    state.{x,y,z,roll,pitch,yaw}             (1,1,1)     float32
    state.gripper                            (1,1,2)     float32
    annotation.human.action.task_description [str]       (B,) list
Server replies with a packed dict: {"actions": (H,7) float32} in the
canonical (x,y,z,roll,pitch,yaw,gripper) order. Gripper is left in the
model's native [0,1] range; the sim side normalises/inverts it.
"""

from __future__ import annotations

import argparse
import logging
import traceback

import numpy as np
import websockets.sync.server
from websockets.exceptions import ConnectionClosed

import msgpack_numpy

ACTION_ORDER = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
FEATURE_KEYS = ("base_image", "wrist_image", "language", "language_mask", "language_len", "state_features")


def make_policy(model_path: str, device: str, strict: bool, with_features: bool):
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

    if with_features:
        from feature_policy import Gr00tFeaturePolicy

        base = Gr00tFeaturePolicy(
            embodiment_tag="LIBERO_PANDA", model_path=model_path, device=device, strict=strict
        )
    else:
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        base = Gr00tPolicy(
            embodiment_tag="LIBERO_PANDA", model_path=model_path, device=device, strict=strict
        )
    return Gr00tSimPolicyWrapper(base, strict=strict)


def infer(policy, obs: dict) -> dict:
    action, info = policy.get_action(obs)  # {"action.x": (1,H,1), ...}
    cols = [np.asarray(action[f"action.{k}"], dtype=np.float32) for k in ACTION_ORDER]
    chunk = np.concatenate(cols, axis=-1)  # (1, H, 7)
    result = {"actions": np.ascontiguousarray(chunk[0], dtype=np.float32)}
    for key in FEATURE_KEYS:
        if key in info:
            result[key] = info[key]
    return result


class GrootWebsocketServer:
    def __init__(self, policy, host: str, port: int, metadata: dict):
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata

    def serve_forever(self) -> None:
        with websockets.sync.server.serve(
            self._handler, self._host, self._port, compression=None, max_size=None
        ) as server:
            logging.info("GR00T websocket server ready on ws://%s:%d", self._host, self._port)
            server.serve_forever()

    def _handler(self, ws) -> None:
        packer = msgpack_numpy.Packer()
        ws.send(packer.pack(self._metadata))
        logging.info("client connected: %s", ws.remote_address)
        while True:
            try:
                msg = ws.recv()
            except ConnectionClosed:
                logging.info("client disconnected")
                return
            try:
                obs = msgpack_numpy.unpackb(msg)
                result = infer(self._policy, obs)
                ws.send(packer.pack(result))
            except Exception:  # noqa: BLE001
                tb = traceback.format_exc()
                logging.error("inference error:\n%s", tb)
                ws.send(tb)  # a str reply makes the openpi client raise RuntimeError


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path",
        default="checkpoints/GR00T-N1.7-LIBERO/libero_10",
        help="path to the libero_10 checkpoint sub-folder",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-strict", action="store_true", help="disable GR00T input/output validation")
    p.add_argument(
        "--with-features",
        action="store_true",
        help="also return the layer-16 backbone hidden states (base_image / wrist_image / language / state_features)",
    )
    args = p.parse_args()

    logging.info(
        "loading GR00T policy from %s ... (features=%s)", args.model_path, args.with_features
    )
    policy = make_policy(
        args.model_path, args.device, strict=not args.no_strict, with_features=args.with_features
    )
    logging.info("policy loaded.")

    meta = {
        "policy": "gr00t-n1.7-libero_10",
        "action_order": list(ACTION_ORDER),
        "with_features": bool(args.with_features),
    }
    GrootWebsocketServer(policy, args.host, args.port, meta).serve_forever()


if __name__ == "__main__":
    main()
