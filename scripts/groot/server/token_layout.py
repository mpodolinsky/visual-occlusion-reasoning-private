"""Pure token-layout helpers for splitting GR00T backbone features.

Cosmos-Reason2-2B / Qwen3-VL sequence for one LIBERO step (batch 1):

    <|im_start|>user\\n <|vision_start|> [64x IMAGE] <|vision_end|>
    <|vision_start|> [64x IMAGE] <|vision_end|> {instruction} <|im_end|>\\n

No torch / numpy here so it is unit-testable without the GR00T venv.
"""

from __future__ import annotations

IMAGE_TOKEN_ID = 151655
VISION_START_ID = 151652
VISION_END_ID = 151653
IM_END_ID = 151645

HIDDEN = 2048
IMG_TOKENS = 64          # 256px / patch 16 -> 16x16 -> 2x2 merge -> 8x8 per camera
LANG_MAX = 200           # instruction tokens zero-padded to this (matches pi0.5 SAVE-A)
STATE_FEATURE_DIM = 1536


def image_runs(input_ids: list[int]) -> list[tuple[int, int]]:
    """Contiguous [start, end) runs of image tokens, in order."""
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(input_ids)
    while i < n:
        if input_ids[i] == IMAGE_TOKEN_ID:
            j = i
            while j < n and input_ids[j] == IMAGE_TOKEN_ID:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def instruction_span(input_ids: list[int]) -> tuple[int, int]:
    """[start, end) of the instruction tokens: after the last <|vision_end|>,
    up to the next <|im_end|> (or end of sequence)."""
    last_vision_end = max(i for i, t in enumerate(input_ids) if t == VISION_END_ID)
    start = last_vision_end + 1
    end = len(input_ids)
    for i in range(start, len(input_ids)):
        if input_ids[i] == IM_END_ID:
            end = i
            break
    return start, end
