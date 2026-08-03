#!/usr/bin/env python3
"""Serves pi05_libero exactly like scripts/serve_policy.py, but each infer()
response also carries the frozen backbone's raw prefix token groups, for
training the perception-uncertainty probe (see scripts/perception_probe/README.md).

Strict superset of the default server: "actions" and "policy_timing" are
computed identically, so anything already talking to the plain server (e.g.
scripts/evaluation/eval_pi05_libero.py) keeps working unchanged against this
one too.

Run with the openpi submodule's own venv:
    submodules/openpi/.venv/bin/python scripts/perception_probe/serve_pi05_with_features.py
"""

from __future__ import annotations

import logging
from pathlib import Path
import socket
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "submodules" / "openpi" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import flax.nnx as nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.policies import policy as _policy  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from openpi_client import base_policy as _base_policy  # noqa: E402

from pi0_features import sample_actions_with_features  # noqa: E402

DEFAULT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
DEFAULT_CONFIG = "pi05_libero"
PORT = 8000

# Fixed by LiberoInputs (submodules/openpi/src/openpi/policies/libero_policy.py):
# images are always [base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb], each
# resized to 224x224 -> 16x16 SigLIP patches. right_wrist is always-zero
# padding (image_mask=False for non-FAST models) so we don't bother returning it.
NUM_IMAGE_TOKENS = 256
NUM_CAMERAS = 3
LANG_START = NUM_IMAGE_TOKENS * NUM_CAMERAS


class FeaturePolicy(_base_policy.BasePolicy):
    """Wraps a Policy; infer() adds a "prefix_features" key with raw per-token
    representations from the frozen VLM prefix (pre-pooling, pre-action-expert).

    Computes actions and prefix features in a single fused forward pass via
    pi0_features.sample_actions_with_features, instead of running the backbone
    twice (once for actions, once more to recover the prefix hidden states).
    Bypasses Policy.infer() for the action path so Policy/policy.py stays
    completely unmodified; the small amount of its plumbing still needed
    (input/output transforms, sample_kwargs) is read directly off `policy`.
    """

    def __init__(self, policy: _policy.Policy):
        if policy._is_pytorch_model:  # noqa: SLF001
            raise NotImplementedError("FeaturePolicy only implements the JAX/nnx path.")
        self._policy = policy
        self._model = policy._model  # noqa: SLF001
        self._input_transform = policy._input_transform  # noqa: SLF001
        self._output_transform = policy._output_transform  # noqa: SLF001
        self._num_steps = policy._sample_kwargs.get("num_steps", 10)  # noqa: SLF001

        # Policy.infer() is bypassed entirely for the action path (see class
        # docstring), so its internal `_rng` stream is never advanced by us;
        # own an independent one instead.
        self._rng = jax.random.key(0)

        # Freeze the (never-mutated) model state once and JIT the fused
        # sample_actions_with_features call, mirroring how Policy itself JITs
        # sample_actions (nnx_utils.module_jit). Un-jitted eager calls pay
        # per-op dispatch overhead and can pick different XLA kernel fusions
        # than a compiled path; compiling this closes that gap.
        graphdef, state = nnx.split(self._model)
        self._state = state

        def _sample(state: nnx.State, rng, observation: _model.Observation, *, num_steps: int):
            model = nnx.merge(graphdef, state)
            return sample_actions_with_features(model, rng, observation, num_steps=num_steps)

        self._jitted_sample = jax.jit(_sample, static_argnames=("num_steps",))

    @property
    def metadata(self) -> dict:
        return self._policy.metadata

    def infer(self, obs: dict) -> dict:
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        start_time = time.monotonic()
        # actions, prefix_out, prefix_mask are all still batched: (1, ...).
        actions, prefix_out, prefix_mask = self._jitted_sample(
            self._state, sample_rng, observation, num_steps=self._num_steps
        )
        model_time = time.monotonic() - start_time

        outputs = {"state": inputs["state"], "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}

        outputs["prefix_features"] = {
            "base_image": np.asarray(prefix_out[0, 0:NUM_IMAGE_TOKENS, :], dtype=np.float16),
            "wrist_image": np.asarray(
                prefix_out[0, NUM_IMAGE_TOKENS : 2 * NUM_IMAGE_TOKENS, :], dtype=np.float16
            ),
            "language": np.asarray(prefix_out[0, LANG_START:, :], dtype=np.float16),
            "language_mask": np.asarray(prefix_mask[0, LANG_START:], dtype=np.bool_),
        }
        return outputs


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    policy = _policy_config.create_trained_policy(_config.get_config(DEFAULT_CONFIG), DEFAULT_CHECKPOINT)
    feature_policy = FeaturePolicy(policy)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating feature-serving server (host: %s, ip: %s, port: %d)", hostname, local_ip, PORT)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=feature_policy,
        host="0.0.0.0",
        port=PORT,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
