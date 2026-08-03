#!/usr/bin/env python3
"""Ad-hoc VQA queries against the exact pi0.5 weights served for LIBERO.

pi0.5 (as implemented in submodules/openpi) has no built-in subtask-prediction
head at inference time -- `Pi0.sample_actions` is pure flow matching over a
SigLIP+Gemma "prefix" embedding, with no autoregressive text decoding path.
The Gemma backbone's vocab projection is weight-tied to its input embedding
table (`gemma.Embedder.decode`) but was not exposed on `gemma.Module`; we
patched that in (see `Module.decode` in
submodules/openpi/src/openpi/models/gemma.py) and implement the incremental
decode loop here, entirely outside the pi0.5-specific action-expert code path.

This loads the pi05_libero checkpoint in-process (not through the websocket
server, since the server's protocol only exposes `infer` for actions) using
the SAME weights and preprocessing the eval scripts use. It is a separate
process/GPU allocation from any websocket server you have running.

Run with the openpi submodule's own venv, which has jax/flax/openpi installed:
    submodules/openpi/.venv/bin/python scripts/vqa/pi05_vqa.py \
        --image path/to/frame.png --wrist-image path/to/wrist.png \
        --task "pick up the black bowl and place it in the tray" \
        --question "What should the robot do right now?"
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENPI_SRC = REPO_ROOT / "submodules" / "openpi" / "src"
sys.path.insert(0, str(OPENPI_SRC))

import einops  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.models.pi0 import make_attn_mask  # noqa: E402
from openpi.policies import policy as _policy  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.shared import download as _download  # noqa: E402
from openpi.training import config as _config  # noqa: E402

DEFAULT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
DEFAULT_CONFIG = "pi05_libero"


def load_pi05_libero_policy(
    config_name: str = DEFAULT_CONFIG, checkpoint_dir: str = DEFAULT_CHECKPOINT
) -> _policy.Policy:
    """Loads the exact same weights/config used to serve pi05_libero over the websocket."""
    return _policy_config.create_trained_policy(_config.get_config(config_name), checkpoint_dir)


@dataclasses.dataclass
class VQAResult:
    text: str
    token_ids: list[int]


class Pi05VQA:
    """Autoregressive text generation through pi0.5's PaliGemma backbone.

    Bypasses the flow-matching action expert entirely. The vision (SigLIP) and
    language (Gemma) weights queried here are identical to the ones conditioning
    pi0.5's action predictions -- this is genuinely "what pi0.5 itself sees",
    not an independent copy of PaliGemma.
    """

    def __init__(self, policy: _policy.Policy, max_prompt_tokens: int | None = None):
        if policy._is_pytorch_model:  # noqa: SLF001
            raise NotImplementedError("Pi05VQA only implements the JAX/nnx sample path.")
        self._policy = policy
        self._model = policy._model  # noqa: SLF001
        self._max_prompt_tokens = max_prompt_tokens or self._model.max_token_len

        tokenizer_path = _download.maybe_download(
            "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
        )
        import sentencepiece

        with tokenizer_path.open("rb") as f:
            self._sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def _build_observation(self, obs: dict, question: str) -> _model.Observation:
        """Runs the real LIBERO input pipeline, but tokenizes `question` instead of the task prompt."""
        inputs = dict(obs)
        inputs["prompt"] = question
        inputs = self._policy._input_transform(inputs)  # noqa: SLF001
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)
        return _model.preprocess_observation(None, observation, train=False)

    def ask(
        self,
        obs: dict,
        question: str,
        *,
        max_new_tokens: int = 40,
        temperature: float = 0.0,
        rng: jax.Array | None = None,
    ) -> VQAResult:
        """Generates a free-form text answer conditioned on the current camera images + `question`.

        `obs` must contain the same keys as `Policy.infer` expects for LIBERO:
        "observation/state", "observation/image", "observation/wrist_image".
        """
        model = self._model
        observation = self._build_observation(obs, question)

        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefix_outputs, kv_cache = model.PaliGemma.llm(
            [prefix_tokens, None], mask=attn_mask, positions=positions
        )
        prefix_out = prefix_outputs[0]

        num_valid = int(jnp.sum(prefix_mask, axis=-1)[0])
        hidden = prefix_out[:, num_valid - 1 : num_valid, :]

        # The prefix KV cache is kept frozen (exactly as sample_actions does for the flow-matching
        # suffix). Rather than repeatedly growing and re-consuming a returned cache -- a code path
        # nothing else in openpi exercises -- each step recomputes attention over the full
        # generated-so-far suffix against that frozen cache, with a causal mask among suffix tokens.
        # Slightly more compute per step, but only exercises the tested call pattern.
        eos_id = self._sp.eos_id()
        generated: list[int] = []
        for step in range(max_new_tokens):
            logits = model.PaliGemma.llm(hidden, method="decode")[:, 0, :]
            if temperature > 0.0:
                step_rng = jax.random.fold_in(rng if rng is not None else jax.random.key(0), step)
                next_id = jax.random.categorical(step_rng, logits / temperature, axis=-1)
            else:
                next_id = jnp.argmax(logits, axis=-1)
            token_id = int(next_id[0])
            if token_id == eos_id:
                break
            generated.append(token_id)

            # Re-embed and re-attend over every token generated so far (expert slot 0, same
            # weights as the prefix), against the frozen prefix kv_cache, with a causal mask
            # among the suffix tokens themselves.
            suffix_ids = jnp.asarray(generated, dtype=jnp.int32)[None, :]  # (1, t)
            suffix_tokens = model.PaliGemma.llm(suffix_ids, method="embed")
            suffix_len = suffix_tokens.shape[1]
            suffix_positions = num_valid + jnp.arange(suffix_len)[None, :]
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_len)
            suffix_attn_mask = make_attn_mask(
                jnp.ones((1, suffix_len), dtype=jnp.bool_), jnp.ones((suffix_len,), dtype=jnp.bool_)
            )
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            outputs, _ = model.PaliGemma.llm(
                [suffix_tokens, None], mask=full_attn_mask, positions=suffix_positions, kv_cache=kv_cache
            )
            hidden = outputs[0][:, -1:, :]

        text = self._sp.decode(generated)
        return VQAResult(text=text, token_ids=generated)


def _load_image(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Third-person (agentview) RGB frame.")
    parser.add_argument(
        "--wrist-image", type=Path, default=None, help="Wrist camera RGB frame (reuses --image if omitted)."
    )
    parser.add_argument("--task", default="pick up the black bowl and place it in the tray")
    parser.add_argument("--question", default="What should the robot do right now?")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_image = _load_image(args.image)
    wrist_image = _load_image(args.wrist_image) if args.wrist_image else base_image

    policy = load_pi05_libero_policy(args.config, args.checkpoint)
    vqa = Pi05VQA(policy)

    obs = {
        "observation/state": np.zeros(8, dtype=np.float32),
        "observation/image": base_image,
        "observation/wrist_image": wrist_image,
        "prompt": args.task,
    }
    # `question` fully replaces `prompt` for tokenization inside ask(); `obs["prompt"]`
    # above is unused but kept so `obs` matches the shape Policy.infer() expects.
    prompted_question = f"Task: {args.task}\nQuestion: {args.question}\nAnswer:"
    result = vqa.ask(
        obs, prompted_question, max_new_tokens=args.max_new_tokens, temperature=args.temperature
    )
    print(f"Q: {prompted_question}")
    print(f"A: {result.text!r}")
    print(f"tokens: {result.token_ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
