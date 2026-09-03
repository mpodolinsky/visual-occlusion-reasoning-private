"""Agentview video: frame index == control_step, fps == control Hz (20)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from constants import CONTROL_HZ


def write_video(path: Path, frames: Sequence[np.ndarray], fps: float = CONTROL_HZ) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [np.ascontiguousarray(x) for x in frames]
    if not arrays:
        raise ValueError("no frames to write")
    height, width = arrays[0].shape[:2]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        for frame in arrays:
            if frame.shape[0] != height or frame.shape[1] != width:
                raise ValueError(f"frame size {frame.shape[:2]} != {(height, width)}")
            proc.stdin.write(frame.astype(np.uint8).tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"FFmpeg failed: {err.strip()}")
        return path
    try:
        import imageio

        imageio.mimwrite(path, arrays, fps=float(fps))
        return path
    except ImportError:
        pass
    import cv2

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    for frame in arrays:
        writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()
    return path
