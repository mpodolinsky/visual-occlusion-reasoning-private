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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "submodules" / "openpi" / "src"))

import flax.nnx as nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.models.pi0 import make_attn_mask  # noqa: E402
from openpi.policies import policy as _policy  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from openpi_client import base_policy as _base_policy  # noqa: E402

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
    """

    def __init__(self, policy: _policy.Policy):
        if policy._is_pytorch_model:  # noqa: SLF001
            raise NotImplementedError("FeaturePolicy only implements the JAX/nnx path.")
        self._policy = policy
        self._model = policy._model  # noqa: SLF001

        # Freeze the (never-mutated) model state once and JIT the prefix forward
        # pass, mirroring how Policy itself JITs sample_actions (nnx_utils.module_jit).
        # Un-jitted eager calls pay per-op dispatch overhead and can pick different
        # XLA kernel fusions than the compiled sample_actions path; compiling this
        # closes that gap and removes the eager-dispatch cost from every inference call.
        graphdef, state = nnx.split(self._model)
        self._prefix_state = state

        def _prefix_forward(state: nnx.State, observation: _model.Observation):
            model = nnx.merge(graphdef, state)
            prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
            attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            prefix_outputs, _ = model.PaliGemma.llm(
                [prefix_tokens, None], mask=attn_mask, positions=positions
            )
            return prefix_outputs[0], prefix_mask

        self._jitted_prefix_forward = jax.jit(_prefix_forward)

    @property
    def metadata(self) -> dict:
        return self._policy.metadata

    def infer(self, obs: dict) -> dict:
        outputs = self._policy.infer(obs)  # unchanged action path

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._policy._input_transform(inputs)  # noqa: SLF001
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)
        observation = _model.preprocess_observation(None, observation, train=False)

        # prefix_out: (1, num_tokens, 2048)
        prefix_out, prefix_mask = self._jitted_prefix_forward(self._prefix_state, observation)

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
