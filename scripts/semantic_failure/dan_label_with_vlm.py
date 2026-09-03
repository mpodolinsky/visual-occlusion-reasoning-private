"""Dan's Gemini failure localizer (liberox-evals analysis/label_with_vlm.py).

Adapted locally with the same prompts and two-pass onset logic:
coarse full-video timestamp, then ±3s 1-FPS frame refine.
Source: https://github.com/dtl184/liberox-evals
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import time
from abc import ABC, abstractmethod

import imageio.v2 as imageio

TAXONOMY = """\
- grasp_failure: the robot attempts to grasp the relevant object but does not successfully acquire it, including repeated unsuccessful grasp attempts
- stuck_or_no_progress: the robot becomes stuck, freezes, repeatedly executes ineffective behavior, or otherwise stops making meaningful progress toward the task
- placement_or_insertion_failure: the robot reaches the relevant target but fails the required placement, insertion, position, or object orientation
- unstable_or_dangerous_behavior: the robot exhibits unstable, erratic, unexpected, or potentially dangerous motion that prevents successful task completion
- object_displacement: the robot unintentionally knocks over, pushes away, drops, or otherwise displaces an object in a way that contributes to task failure
- wrong_object_or_target: the robot manipulates the wrong object or moves the correct object toward the wrong destination
- timeout_or_insufficient_progress: no discrete mistake is clearly identifiable and the robot simply does not complete the task before the episode ends
- other: the observed failure does not fit one of the categories above; explain the failure clearly
"""

COARSE_PROMPT_TEMPLATE = """You are reviewing the COMPLETE video of a failed robot manipulation rollout in LIBERO-X.

The video begins at rollout time 0.0 seconds and has a duration of approximately {duration_seconds:.2f} seconds.

Task instruction:
"{task_desc}"

The simulator reports that this goal predicate was NOT satisfied at the end of the episode:
{failing_predicate}

Additional simulator information:
{detail}

The simulator predicate tells you WHAT goal condition was unsatisfied at the end. It does NOT tell you when the behavior that caused the failure began.

Your job is to determine:

1. the primary visible failure mode,
2. when the failure behavior begins,
3. why the episode failed,
4. and what corrective action should be taken.

FAILURE ONSET DEFINITION

Use exactly one of the following onset types:

obvious_mistake:
 Use this when there is an identifiable task-relevant mistake such as
 dropping an object, knocking an object away, manipulating the wrong
 object, making a failed placement, or otherwise performing an action
 that contributes to the eventual failure.

 The onset should be the EARLIEST point where the mistake responsible
 for the eventual failure becomes apparent.

operator_intervention:
 Use this when there is not one discrete mistake, but the robot reaches
 a point where a reasonable human operator would decide that the policy
 needs assistance.

 Examples include repeatedly attempting an ineffective action, becoming
 stuck, moving erratically, or clearly ceasing to make useful progress.

timeout:
 Use this when there is no defensible earlier failure point. If the robot
 continues behaving plausibly but simply does not finish before the time
 limit, the failure onset is the END of the rollout.

IMPORTANT RULES

- Do NOT treat the goal predicate being false early in the rollout as failure.
 Most goal predicates are naturally false until the robot completes the task.

- Do NOT label a temporary mistake as failure onset if the robot later
 successfully recovers from that mistake.

- Use the earliest mistake or intervention point that actually explains the
 eventual failed outcome.

- Do not use hindsight to label normal task execution as failure merely because
 you know the episode eventually fails.

- If there is no identifiable earlier point, use onset_type="timeout" and set
 failure_onset_seconds to approximately {duration_seconds:.2f}.

FAILURE TAXONOMY

Choose exactly one:

{taxonomy}

RECOVERY ACTION

The recovery action should describe what the robot should do at or immediately
after the identified failure onset in order to recover.

It should identify:
1. what object or mechanism to interact with,
2. what corrective action to perform,
3. what condition should be achieved before continuing.

Return ONLY a JSON object with this structure:

{{
 "failure_mode": " ",
 "onset_type": "obvious_mistake|operator_intervention|timeout",
 "failure_onset_seconds":,
 "failure_window_start_seconds":,
 "failure_window_end_seconds":,
 "confidence": "high|medium|low",
 "failure_reason": " ",
 "recovery_action": " ",
 "justification": " "
}}
"""

REFINE_PROMPT_TEMPLATE = """You are reviewing a TEMPORALLY MAGNIFIED clip from a failed robot rollout.

The complete rollout was already reviewed by another pass of the same VLM.

The coarse analysis identified:

Failure mode:
{failure_mode}

Failure onset type:
{onset_type}

Failure reason:
{failure_reason}

Coarse onset estimate in the original rollout:
{coarse_seconds:.2f} seconds

Task instruction:
"{task_desc}"

Failed simulator predicate:
{failing_predicate}

TEMPORAL MAPPING

This clip has been deliberately slowed down.

Each SECOND of this clip corresponds to exactly ONE FRAME from the original
rollout.

Therefore:

 refined clip second 0 = original rollout frame {start_frame}
 refined clip second 1 = original rollout frame {start_frame_plus_one}
 refined clip second 2 = original rollout frame {start_frame_plus_two}
 ...

The clip contains {num_frames} original rollout frames.

Your job is ONLY to refine the temporal location of the failure.

Find the EARLIEST frame in this clip where the failure identified above becomes
visibly apparent or where operator intervention becomes justified.

Remember:

- Do not select a temporary mistake if the robot recovers from it.
- Do not select a frame merely because the task is not complete yet.
- Select the first frame associated with the behavior responsible for the
 eventual failure.
- If the coarse event is not actually visible in this clip, return null.

Return ONLY:

{{
 "refined_second":,
 "confidence": "high|medium|low",
 "justification": " "
}}
"""


def parse_json_response(text: str) -> dict:
    if not text:
        raise ValueError("Empty VLM response")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in VLM response: {text[:300]}")
    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(cleaned[start:])
        return parsed
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse VLM JSON response: {text[:500]}") from exc


def get_video_metadata(video_path):
    reader = imageio.get_reader(str(video_path))
    try:
        meta = reader.get_meta_data()
        fps = float(meta.get("fps") or 20.0)
        if fps <= 0:
            raise ValueError(f"Invalid FPS for video: {video_path}")
        num_frames = int(reader.count_frames())
    finally:
        reader.close()
    if num_frames == 0:
        raise ValueError(f"Video contains no frames: {video_path}")
    duration_seconds = (num_frames - 1) / fps if num_frames > 1 else 0.0
    return {"num_frames": num_frames, "fps": fps, "duration_seconds": duration_seconds}


def make_refinement_clip(video_path, output_path, center_frame, radius_frames):
    reader = imageio.get_reader(str(video_path))
    try:
        total_frames = int(reader.count_frames())
        start_frame = max(0, center_frame - radius_frames)
        end_frame = min(total_frames - 1, center_frame + radius_frames)
        frames = [reader.get_data(i) for i in range(start_frame, end_frame + 1)]
    finally:
        reader.close()
    imageio.mimwrite(str(output_path), frames, fps=1)
    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "num_frames": len(frames),
    }


def clamp(value, low, high):
    return max(low, min(high, value))


def safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_timestamp(seconds):
    if seconds is None:
        return ""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remaining = seconds - 60 * minutes
    return f"{minutes:02d}:{remaining:04.1f}"


class VLMBackend(ABC):
    def __init__(self, model, max_retries=4):
        self.model = model
        self.max_retries = max_retries

    @abstractmethod
    def generate_video(self, video_path, prompt):
        raise NotImplementedError


class GeminiBackend(VLMBackend):
    def __init__(self, model="gemini-3.1-pro-preview", max_retries=4):
        super().__init__(model=model, max_retries=max_retries)
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError("Gemini backend requires google-genai") from exc
        self.client = genai.Client()

    def open_video_session(self, video_path):
        from gemini_session import GeminiVideoSession

        return GeminiVideoSession(
            self.client, self.model, pathlib.Path(video_path), max_retries=self.max_retries
        )

    def _wait_for_file(self, uploaded):
        while True:
            state = getattr(uploaded, "state", None)
            state_name = getattr(state, "name", str(state) if state else "")
            if state_name == "ACTIVE":
                return uploaded
            if state_name == "FAILED":
                raise RuntimeError("Gemini video processing failed")
            time.sleep(2)
            uploaded = self.client.files.get(name=uploaded.name)

    @staticmethod
    def _extract_usage(response):
        from gemini_session import extract_usage

        return extract_usage(response)

    def _generate_once(self, video_path, prompt):
        uploaded = None
        try:
            uploaded = self.client.files.upload(file=str(video_path))
            uploaded = self._wait_for_file(uploaded)
            response = self.client.models.generate_content(
                model=self.model,
                contents=[uploaded, prompt],
            )
            text = getattr(response, "text", None)
            if not text:
                raise RuntimeError("Gemini returned no text")
            return {"text": text.strip(), "usage": self._extract_usage(response)}
        finally:
            if uploaded is not None:
                try:
                    self.client.files.delete(name=uploaded.name)
                except Exception:
                    pass

    def generate_video(self, video_path, prompt):
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                return self._generate_once(video_path=video_path, prompt=prompt)
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(min(60, 2 ** attempt * 2))
        raise last_exc


BACKENDS = {"gemini": GeminiBackend}


def create_backend(backend_name, model):
    if backend_name not in BACKENDS:
        raise ValueError(f"Unsupported VLM backend: {backend_name}")
    return BACKENDS[backend_name](model=model)


def coarse_prompt(row, metadata) -> str:
    return COARSE_PROMPT_TEMPLATE.format(
        duration_seconds=metadata["duration_seconds"],
        task_desc=row.get("task_desc", ""),
        failing_predicate=row.get("failing_predicate", ""),
        detail=row.get("detail", ""),
        taxonomy=TAXONOMY,
    )


def coarse_label(backend, row, video_path, metadata):
    result = backend.generate_video(video_path=video_path, prompt=coarse_prompt(row, metadata))
    parsed = parse_json_response(result["text"])
    return {"parsed": parsed, "text": result["text"], "usage": result["usage"]}


def coarse_label_on_session(session, row, metadata):
    result = session.ask(coarse_prompt(row, metadata))
    parsed = parse_json_response(result["text"])
    return {"parsed": parsed, "text": result["text"], "usage": result["usage"]}


def refine_failure_onset(backend, row, video_path, metadata, coarse, refine_window_seconds):
    parsed = coarse["parsed"]
    onset_type = str(parsed.get("onset_type", "")).strip()
    duration = metadata["duration_seconds"]
    fps = metadata["fps"]
    num_frames = metadata["num_frames"]
    coarse_seconds = clamp(safe_float(parsed.get("failure_onset_seconds"), duration), 0.0, duration)

    if onset_type == "timeout":
        final_frame = num_frames - 1
        return {
            "frame": final_frame,
            "seconds": duration,
            "refined": False,
            "refinement_text": "",
            "refinement_usage": {"input_tokens": 0, "output_tokens": 0},
            "refinement_confidence": parsed.get("confidence", "low"),
            "refinement_justification": (
                "No earlier failure event identified; using the final frame."
            ),
        }

    coarse_frame = int(clamp(round(coarse_seconds * fps), 0, num_frames - 1))
    radius_frames = max(1, int(round(refine_window_seconds * fps)))
    with tempfile.TemporaryDirectory(prefix="libero10_failure_refine_") as temp_dir:
        refinement_path = pathlib.Path(temp_dir) / "refinement.mp4"
        clip = make_refinement_clip(video_path, refinement_path, coarse_frame, radius_frames)
        prompt = REFINE_PROMPT_TEMPLATE.format(
            failure_mode=parsed.get("failure_mode", "unknown"),
            onset_type=onset_type,
            failure_reason=parsed.get("failure_reason", ""),
            coarse_seconds=coarse_seconds,
            task_desc=row.get("task_desc", ""),
            failing_predicate=row.get("failing_predicate", ""),
            start_frame=clip["start_frame"],
            start_frame_plus_one=clip["start_frame"] + 1,
            start_frame_plus_two=clip["start_frame"] + 2,
            num_frames=clip["num_frames"],
        )
        if hasattr(backend, "ask_other_video"):
            refinement = backend.ask_other_video(refinement_path, prompt)
        else:
            refinement = backend.generate_video(video_path=refinement_path, prompt=prompt)
        refinement_parsed = parse_json_response(refinement["text"])
        refined_second = refinement_parsed.get("refined_second")
        if refined_second is None:
            final_frame = coarse_frame
            refined = False
        else:
            try:
                refined_second = int(round(float(refined_second)))
            except (TypeError, ValueError):
                refined_second = None
            if refined_second is None:
                final_frame = coarse_frame
                refined = False
            else:
                refined_second = int(clamp(refined_second, 0, clip["num_frames"] - 1))
                final_frame = int(clamp(clip["start_frame"] + refined_second, 0, num_frames - 1))
                refined = True
        return {
            "frame": final_frame,
            "seconds": final_frame / fps,
            "refined": refined,
            "refinement_text": refinement["text"],
            "refinement_usage": refinement["usage"],
            "refinement_confidence": refinement_parsed.get("confidence", "low"),
            "refinement_justification": refinement_parsed.get("justification", ""),
        }


def timeout_or_unrefined_temporal(parsed, metadata):
    onset_type = parsed.get("onset_type", "")
    coarse_seconds = clamp(
        safe_float(parsed.get("failure_onset_seconds"), metadata["duration_seconds"]),
        0.0,
        metadata["duration_seconds"],
    )
    if onset_type == "timeout":
        frame = metadata["num_frames"] - 1
        seconds = metadata["duration_seconds"]
    else:
        frame = int(clamp(round(coarse_seconds * metadata["fps"]), 0, metadata["num_frames"] - 1))
        seconds = frame / metadata["fps"]
    return {
        "frame": frame,
        "seconds": seconds,
        "refined": False,
        "refinement_text": "",
        "refinement_usage": {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0},
        "refinement_confidence": "",
        "refinement_justification": "",
    }


def pack_failure_result(metadata, coarse, temporal) -> dict:
    parsed = coarse["parsed"]
    onset_frame = temporal["frame"]
    onset_seconds = temporal["seconds"]
    coarse_usage = coarse["usage"]
    refine_usage = temporal["refinement_usage"]
    return {
        "vlm_failure_mode": parsed.get("failure_mode"),
        "vlm_failure_onset_type": parsed.get("onset_type"),
        "vlm_failure_onset_seconds": onset_seconds,
        "vlm_failure_onset_timestamp": format_timestamp(onset_seconds),
        "vlm_failure_onset_frame": onset_frame,
        "vlm_failure_onset_step": onset_frame,
        "vlm_coarse_failure_onset_seconds": safe_float(parsed.get("failure_onset_seconds")),
        "vlm_failure_window_start_seconds": safe_float(parsed.get("failure_window_start_seconds")),
        "vlm_failure_window_end_seconds": safe_float(parsed.get("failure_window_end_seconds")),
        "vlm_temporal_refined": temporal["refined"],
        "vlm_confidence": parsed.get("confidence"),
        "vlm_temporal_confidence": temporal["refinement_confidence"],
        "vlm_failure_reason": parsed.get("failure_reason"),
        "vlm_recovery_action": parsed.get("recovery_action"),
        "vlm_justification": parsed.get("justification"),
        "vlm_temporal_justification": temporal["refinement_justification"],
        "vlm_video_fps": metadata["fps"],
        "vlm_video_num_frames": metadata["num_frames"],
        "vlm_video_duration_seconds": metadata["duration_seconds"],
        "vlm_raw_response": coarse["text"],
        "vlm_refinement_raw_response": temporal["refinement_text"],
        "vlm_usage": {
            "input_tokens": coarse_usage.get("input_tokens", 0) + refine_usage.get("input_tokens", 0),
            "output_tokens": coarse_usage.get("output_tokens", 0)
            + refine_usage.get("output_tokens", 0),
            "cached_tokens": coarse_usage.get("cached_tokens", 0)
            + refine_usage.get("cached_tokens", 0),
        },
    }


def compact_failure_for_context(vlm: dict) -> dict:
    """Fields the 3s turn may see as context. Never written into semantic phrases."""
    return {
        "failure_mode": vlm.get("vlm_failure_mode"),
        "onset_type": vlm.get("vlm_failure_onset_type"),
        "failure_onset_seconds": vlm.get("vlm_failure_onset_seconds"),
        "failure_onset_frame": vlm.get("vlm_failure_onset_frame"),
        "failure_reason": vlm.get("vlm_failure_reason"),
        "recovery_action": vlm.get("vlm_recovery_action"),
    }


def label_one_on_session(session, row, refine=True, refine_window_seconds=3.0):
    video_path = pathlib.Path(row["video_path"])
    metadata = get_video_metadata(video_path)
    coarse = coarse_label_on_session(session, row, metadata)
    if refine:
        temporal = refine_failure_onset(
            session, row, video_path, metadata, coarse, refine_window_seconds
        )
    else:
        temporal = timeout_or_unrefined_temporal(coarse["parsed"], metadata)
    return pack_failure_result(metadata, coarse, temporal)


def label_one(backend, row, refine=True, refine_window_seconds=3.0):
    video_path = pathlib.Path(row["video_path"])
    metadata = get_video_metadata(video_path)
    coarse = coarse_label(backend, row, video_path, metadata)
    if refine:
        temporal = refine_failure_onset(
            backend, row, video_path, metadata, coarse, refine_window_seconds
        )
    else:
        temporal = timeout_or_unrefined_temporal(coarse["parsed"], metadata)
    return pack_failure_result(metadata, coarse, temporal)

