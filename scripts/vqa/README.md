# pi0.5 VQA probe

`pi05_vqa.py` queries the exact SigLIP+Gemma weights behind the served
`pi05_libero` checkpoint in a VQA style (image + free-text question -> text
answer), bypassing the flow-matching action expert entirely.

## Why this exists

pi0.5, as implemented in `submodules/openpi`, has no subtask-prediction head
at inference time: `Pi0.sample_actions` is pure flow matching over a
SigLIP+Gemma prefix embedding, with no autoregressive text decoding path
wired in anywhere (see the docstring in `pi05_vqa.py` for specifics). The
vocab projection needed for text decoding exists in the Gemma backbone
(`gemma.Embedder.decode`, weight-tied to the input embedding) but wasn't
exposed; we added `Module.decode(...)` in
`submodules/openpi/src/openpi/models/gemma.py` to surface it, and implement
the incremental decode loop here, outside pi0.5's action-specific code path.

## Empirical finding

Free-form generation from `pi05_libero` is **not usable for open-text
subtask/VQA output**. Across prompt phrasings (templated question, or the raw
LIBERO task string alone), the model reliably predicts one highly-confident
first token -- `"Sub"` (~99.7% probability, very low entropy; runner-up
completions are "return"/"adjust"/"move", clearly heading toward
"Subtask: <verb>...") -- then immediately collapses into token IDs in the
254000-256000 range for every subsequent step. That range is exactly where
FAST action tokens get mapped
(`tokenizer.py`'s `_act_tokens_to_paligemma_tokens`:
`vocab_size() - 1 - fast_skip_tokens - bin_id`). The model is not emitting
noise -- it is confidently trying to continue "Sub..." with *encoded action
tokens*, not English words. This is consistent with pi05_libero's
post-training having reinforced predicting FAST-style action continuations
rather than the open "Subtask: pick up X" text pi0.5's paper describes;
whatever general VQA competence the base PaliGemma pretraining had did not
survive fine-tuning on LIBERO.

This was verified with two independent implementations of the decode loop
(one that incrementally grows and re-consumes a KV cache across steps, one
that keeps the prefix cache frozen and recomputes the growing suffix causally
each step -- mirroring the only KV-cache-reuse pattern `sample_actions`
itself actually exercises). Both produced identical output token-for-token,
ruling out a decode-loop bug.

## Running it

This needs the full `openpi` stack (JAX/Flax/nnx), which lives in
`submodules/openpi`'s own environment, not this repo's top-level venv:

```bash
submodules/openpi/.venv/bin/python scripts/vqa/pi05_vqa.py \
  --image path/to/agentview_frame.png \
  --wrist-image path/to/wrist_frame.png \
  --task "pick up the black bowl and place it in the tray" \
  --question "What should the robot do right now?"
```

It loads the checkpoint in-process (not through the websocket server, since
the server's protocol only exposes `infer` for actions), so it needs its own
GPU memory allocation separate from any running `serve_policy.py` process. If
the GPU is already near full (check with `nvidia-smi`), either free memory
first or add `JAX_PLATFORMS=cpu` to run on CPU (slow, but fine for
correctness checks -- that's how the finding above was verified).
