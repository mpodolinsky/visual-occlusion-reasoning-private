"""Standalone fork of Pi0.sample_actions that also surfaces the prefix hidden
states it already computes internally (and normally discards), instead of
running a second, separate forward pass to recover them.

Lives outside submodules/openpi on purpose: it only calls already-public
methods on a Pi0 instance (embed_prefix, embed_suffix, PaliGemma.llm,
action_out_proj), so it doesn't need to subclass or monkeypatch anything,
and a `git submodule update` can never touch or lose it.

Forked from: submodules/openpi/src/openpi/models/pi0.py, Pi0.sample_actions
(lines 217-243 as of the fork). If openpi's sample_actions changes upstream,
this copy will not pick that up automatically -- the numeric verification in
scripts/perception_probe/verify_features.py is the guard against silent drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import einops
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0
    from openpi.shared import array_typing as at


def sample_actions_with_features(
    model: "Pi0",
    rng: "at.KeyArrayLike",
    observation: "_model.Observation",
    *,
    num_steps: int = 10,
    noise: "at.Float[at.Array, 'b ah ad'] | None" = None,
) -> tuple["_model.Actions", "at.Float[at.Array, 'b p 2048']", "at.Bool[at.Array, 'b p']"]:
    """Same computation as Pi0.sample_actions, plus the prefix hidden states.

    Returns (actions, prefix_out, prefix_mask). prefix_out is the literal
    tensor that conditioned the returned actions -- not a recomputation.
    """
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))

    # first fill KV cache with a forward pass of the prefix
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    prefix_outputs, kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )
    # the prefix's hidden states, discarded as `_` in the original sample_actions --
    # this is the one line that differs from the upstream method.
    prefix_out = prefix_outputs[0]

    def step(carry):
        x_t, time = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask_ = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask_, suffix_attn_mask], axis=-1)
        positions_ = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (_prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions_,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])

        return x_t + dt * v_t, time + dt

    def cond(carry):
        _x_t, time = carry
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0, prefix_out, prefix_mask
