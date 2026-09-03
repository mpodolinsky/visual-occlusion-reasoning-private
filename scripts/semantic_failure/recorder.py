"""Write episode dir: 20 Hz agentview + wrist mp4, json/npz."""

from __future__ import annotations

from pathlib import Path

from records import RolloutRecord
from constants import CONTROL_HZ
from serialization import save_rollout
from video import write_video


def save_episode(directory: Path, rollout: RolloutRecord, *, save_video: bool) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    frames = getattr(rollout, "_video_frames", None)
    wrist_frames = getattr(rollout, "_wrist_frames", None)
    if save_video and frames:
        video_path = directory / "rollout.mp4"
        write_video(video_path, frames, fps=CONTROL_HZ)
        rollout.video_path = str(video_path)
    if save_video and wrist_frames:
        wrist_path = directory / "wrist.mp4"
        write_video(wrist_path, wrist_frames, fps=CONTROL_HZ)
        rollout.wrist_video_path = str(wrist_path)
    save_rollout(directory, rollout)
    return directory
