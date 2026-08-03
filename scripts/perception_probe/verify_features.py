#!/usr/bin/env python3
"""One-off numeric verification: confirms the new fused-call prefix features
(sample_actions_with_features) agree with the old, separately-jitted two-pass
extraction (archived in archive/serve_pi05_with_features_two_pass.py) on real
LIBERO observations, before the two-pass code is retired from the live server.

Loads the policy exactly once so both paths run against the identical
in-memory weights -- this isolates "did I implement the fork correctly" from
"do two independent JIT compilations of the same math agree bit-for-bit".

Run with the openpi submodule's own venv:
    submodules/openpi/.venv/bin/python scripts/perception_probe/verify_features.py
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "submodules" / "openpi" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

import flax.nnx as nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.models.pi0 import make_attn_mask  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402

from pi0_features import sample_actions_with_features  # noqa: E402

DEFAULT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
DEFAULT_CONFIG = "pi05_libero"


def old_prefix_forward(model, observation):
    """Verbatim copy of the retired FeaturePolicy._prefix_forward body."""
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    prefix_outputs, _ = model.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
    return prefix_outputs[0], prefix_mask


def load_observation(npz_path: Path) -> dict:
    """Loads a LiberoInputs-shaped observation dumped by a top-level-venv helper
    (this venv has no robosuite/libero, so the raw observation is produced
    out-of-process and handed over as a plain .npz)."""
    with np.load(npz_path, allow_pickle=True) as data:
        return {
            "observation/image": data["image"],
            "observation/wrist_image": data["wrist_image"],
            "observation/state": data["state"],
            "prompt": str(data["prompt"]),
        }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    obs_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/verify_obs.npz")
    logging.info("Loading dumped observation from %s", obs_path)
    raw_obs = load_observation(obs_path)

    logging.info("Loading policy (single instance, shared by both code paths)...")
    policy = _policy_config.create_trained_policy(_config.get_config(DEFAULT_CONFIG), DEFAULT_CHECKPOINT)
    model = policy._model  # noqa: SLF001

    inputs = jax.tree.map(lambda x: x, raw_obs)
    inputs = policy._input_transform(inputs)  # noqa: SLF001
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = _model.Observation.from_dict(inputs)
    observation = _model.preprocess_observation(None, observation, train=False)

    graphdef, state = nnx.split(model)

    logging.info("Running OLD (separately-jitted) two-pass extraction...")

    def _old(state, observation):
        merged = nnx.merge(graphdef, state)
        return old_prefix_forward(merged, observation)

    old_forward = jax.jit(_old)
    prefix_out_old, prefix_mask_old = old_forward(state, observation)

    logging.info("Running NEW fused sample_actions_with_features...")

    def _sample(state, rng, observation, *, num_steps):
        merged = nnx.merge(graphdef, state)
        return sample_actions_with_features(merged, rng, observation, num_steps=num_steps)

    jitted_sample = jax.jit(_sample, static_argnames=("num_steps",))
    rng = jax.random.key(0)
    _actions_new, prefix_out_new, prefix_mask_new = jitted_sample(state, rng, observation, num_steps=10)

    prefix_out_old = np.asarray(prefix_out_old, dtype=np.float32)
    prefix_out_new = np.asarray(prefix_out_new, dtype=np.float32)
    prefix_mask_old = np.asarray(prefix_mask_old)
    prefix_mask_new = np.asarray(prefix_mask_new)

    logging.info("prefix_out shapes: old=%s new=%s", prefix_out_old.shape, prefix_out_new.shape)
    assert np.array_equal(prefix_mask_old, prefix_mask_new), "prefix_mask mismatch"

    max_abs_diff = float(np.max(np.abs(prefix_out_old - prefix_out_new)))
    max_rel_diff = float(
        np.max(np.abs(prefix_out_old - prefix_out_new) / (np.abs(prefix_out_old) + 1e-6))
    )
    logging.info("max_abs_diff=%.6f max_rel_diff=%.6f", max_abs_diff, max_rel_diff)

    np.testing.assert_allclose(prefix_out_old, prefix_out_new, atol=1e-2, rtol=1e-2)
    logging.info("VERIFICATION PASSED: old and new prefix features agree within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
