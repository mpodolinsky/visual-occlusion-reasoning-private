"""Send one synthetic observation to the running GR00T websocket server.

Run inside the Isaac-GR00T venv:
    submodules/Isaac-GR00T/.venv/bin/python scripts/groot/server/smoke_server.py

Ported verbatim from ``16-LIBERO-X-GR00T-ZeroShot/server/smoke_server.py``.
"""

from __future__ import annotations

import argparse

import numpy as np

import msgpack_numpy
import websockets.sync.client


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    packer = msgpack_numpy.Packer()
    obs = {
        "video.image": np.zeros((1, 1, 256, 256, 3), np.uint8),
        "video.wrist_image": np.zeros((1, 1, 256, 256, 3), np.uint8),
        "state.x": np.zeros((1, 1, 1), np.float32),
        "state.y": np.zeros((1, 1, 1), np.float32),
        "state.z": np.zeros((1, 1, 1), np.float32),
        "state.roll": np.zeros((1, 1, 1), np.float32),
        "state.pitch": np.zeros((1, 1, 1), np.float32),
        "state.yaw": np.zeros((1, 1, 1), np.float32),
        "state.gripper": np.zeros((1, 1, 2), np.float32),
        "annotation.human.action.task_description": ["pick up the black bowl"],
    }
    with websockets.sync.client.connect(
        f"ws://{args.host}:{args.port}", compression=None, max_size=None
    ) as ws:
        meta = msgpack_numpy.unpackb(ws.recv())
        print("metadata:", meta)
        ws.send(packer.pack(obs))
        resp = ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(resp)
        out = msgpack_numpy.unpackb(resp)
        act = np.asarray(out["actions"])
        print("actions shape:", act.shape, "dtype:", act.dtype)
        print("finite:", bool(np.isfinite(act).all()), "absmax:", float(np.abs(act).max()))
        assert act.ndim == 2 and act.shape[1] == 7, act.shape
        print("SERVER SMOKE OK")


if __name__ == "__main__":
    main()
