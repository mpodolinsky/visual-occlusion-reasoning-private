"""Pure tests for the GR00T backbone-feature token split. No torch / GR00T venv."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from token_layout import (  # noqa: E402
    IMAGE_TOKEN_ID,
    IM_END_ID,
    IMG_TOKENS,
    VISION_END_ID,
    VISION_START_ID,
    image_runs,
    instruction_span,
)

IM_START = 151644
NL = 198  # arbitrary text token id stand-in


def _seq(instruction_ids: list[int]) -> list[int]:
    """Rebuild the exact GR00T chat layout with two 64-token image blocks."""
    img = [IMAGE_TOKEN_ID] * IMG_TOKENS
    return (
        [IM_START, NL, VISION_START_ID]
        + img
        + [VISION_END_ID, VISION_START_ID]
        + img
        + [VISION_END_ID]
        + instruction_ids
        + [IM_END_ID, NL]
    )


class TokenLayoutTest(unittest.TestCase):
    def test_image_runs(self) -> None:
        seq = _seq([10, 11, 12])
        runs = image_runs(seq)
        self.assertEqual(len(runs), 2)
        (b0, b1), (w0, w1) = runs
        self.assertEqual(b1 - b0, 64)
        self.assertEqual(w1 - w0, 64)
        self.assertEqual(b0, 3)          # after <|im_start|>, \n, <|vision_start|>
        self.assertEqual(w0, 3 + 64 + 2)  # + first block + <|vision_end|><|vision_start|>

    def test_instruction_span(self) -> None:
        instr = [10, 11, 12, 13, 14]
        seq = _seq(instr)
        s, e = instruction_span(seq)
        self.assertEqual(seq[s:e], instr)
        self.assertEqual(e - s, 5)

    def test_instruction_span_single_token(self) -> None:
        seq = _seq([99])
        s, e = instruction_span(seq)
        self.assertEqual(seq[s:e], [99])

    def test_no_trailing_im_end(self) -> None:
        # defensive: span runs to end of sequence if <|im_end|> is absent
        img = [IMAGE_TOKEN_ID] * IMG_TOKENS
        seq = [VISION_START_ID] + img + [VISION_END_ID] + [1, 2, 3]
        s, e = instruction_span(seq)
        self.assertEqual(seq[s:e], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
