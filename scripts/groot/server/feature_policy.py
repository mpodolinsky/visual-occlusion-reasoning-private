"""Gr00tPolicy subclass that also returns the frozen VLM backbone features.

`Gr00tPolicy._get_action` runs the backbone + action head but keeps only
`action_pred`. This subclass re-implements the same pipeline, splits the
layer-`select_layer` backbone hidden states (`backbone_features`, the pi0.5
SAVE-A analog) by token type, and hands them back in the `info` dict:

    base_image      (64, 2048)   float16   first camera's merged patches
    wrist_image     (64, 2048)   float16   second camera's merged patches
    language        (200, 2048)  float16   instruction tokens, zero-padded
    language_mask   (200,)       bool      real vs pad
    language_len    ()           int32     real instruction token count
    state_features  (1536,)      float16   action head's embedded proprio

Raw layer-16 residual stream, before the action head's `vlln` + VL
self-attention (those are action-specialised; this is the frozen backbone).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

from token_layout import HIDDEN, IMG_TOKENS, LANG_MAX, image_runs, instruction_span


class Gr00tFeaturePolicy(Gr00tPolicy):
    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        unbatched = self._unbatch_observation(observation)
        assert len(unbatched) == 1, "feature capture assumes batch size 1"

        states = []
        processed_inputs = []
        for obs in unbatched:
            vla = self._to_vla_step_data(obs)
            states.append(vla.states)
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla}]
            processed_inputs.append(self.processor(messages))

        collated = self.collate_fn(processed_inputs)
        collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
        inputs = collated["inputs"] if "inputs" in collated else collated

        with torch.inference_mode():
            backbone_inputs, action_inputs = self.model.prepare_input(inputs)
            backbone_outputs = self.model.backbone(backbone_inputs)
            action_outputs = self.model.action_head.get_action(
                backbone_outputs, action_inputs, options
            )

        normalized_action = action_outputs["action_pred"].float()

        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)
        unnormalized = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )
        action = {k: v.astype(np.float32) for k, v in unnormalized.items()}

        info = self._extract_features(backbone_inputs, backbone_outputs, action_outputs)
        return action, info

    @staticmethod
    def _extract_features(backbone_inputs, backbone_outputs, action_outputs) -> dict[str, Any]:
        ids = backbone_inputs["input_ids"][0].tolist()
        feats = backbone_outputs["backbone_features"][0].float().cpu().numpy()  # [L, 2048]

        runs = image_runs(ids)
        assert len(runs) >= 2, f"expected >=2 image-token runs, got {len(runs)}"
        (b0, b1), (w0, w1) = runs[0], runs[1]
        base = feats[b0:b1]
        wrist = feats[w0:w1]
        assert base.shape == (IMG_TOKENS, HIDDEN), base.shape
        assert wrist.shape == (IMG_TOKENS, HIDDEN), wrist.shape

        s, e = instruction_span(ids)
        instr = feats[s:e]                       # [n_lang, 2048]
        n_lang = int(instr.shape[0])
        lang = np.zeros((LANG_MAX, HIDDEN), dtype=np.float16)
        mask = np.zeros((LANG_MAX,), dtype=np.bool_)
        take = min(n_lang, LANG_MAX)
        lang[:take] = instr[:take].astype(np.float16)
        mask[:take] = True

        state_feats = action_outputs["state_features"][0].float().cpu().numpy()  # [S, 1536]
        state_feats = np.ascontiguousarray(state_feats.reshape(-1)[-1536:], dtype=np.float16)

        return {
            "base_image": np.ascontiguousarray(base, dtype=np.float16),
            "wrist_image": np.ascontiguousarray(wrist, dtype=np.float16),
            "language": lang,
            "language_mask": mask,
            "language_len": np.int32(n_lang),
            "state_features": state_feats,
        }
