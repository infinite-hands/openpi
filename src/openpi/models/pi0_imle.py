"""IMLE-VLA: pi0.5 with its 10-step flow-matching head replaced by a single-step cIMLE generator.

Hosseinkhani et al., "IMLE-VLA: Fast Single-Step Action Generation for Vision-Language-Action
Policies" (arXiv:2609.10915v1). The VLM backbone is untouched -- the change is confined to how the
action expert is driven and to the objective it is trained under.
"""

import einops
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
from openpi.shared import array_typing as at

# The generator reads the action expert at the pure-noise end of the flow, where pi0.5 already
# expects a N(0, I) input -- which is what makes initializing from a pi0.5 checkpoint meaningful.
IMLE_TIMESTEP = 1.0


class Pi0IMLE(Pi0):
    """pi0.5 whose action head is a single-step conditional generator G(f_VLM(o), z) -> A."""

    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        self.imle_sample_factor = config.imle_sample_factor

    def _prefix_cache(self, observation: _model.Observation):
        """One backbone pass, shared by every candidate: the paper's f_VLM(o) computed once."""
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
        return kv_cache, prefix_mask

    def _generate(self, observation, z, kv_cache, prefix_mask) -> _model.Actions:
        """G(f_VLM(o), z): one action-expert pass over the cached prefix, no integration loop.

        The expert still emits pi0.5's velocity field, so the chunk is read off as a single Euler
        step from t=1 to t=0. That parameterization is what carries the pretrained head over: at
        initialization G already reproduces a one-step approximation of the flow it replaces, and
        cIMLE fine-tunes from there rather than from noise.
        """
        batch_size = z.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, z, jnp.full((batch_size,), IMLE_TIMESTEP)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return z - velocity

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng = jax.random.split(rng)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        kv_cache, prefix_mask = self._prefix_cache(observation)

        # Assignment step (Eq. 3), gradient-free: m candidates per sample, nearest one wins.
        latents = jax.random.normal(noise_rng, (self.imle_sample_factor, *actions.shape))
        candidates = jax.lax.stop_gradient(
            jnp.stack([self._generate(observation, latents[j], kv_cache, prefix_mask)
                       for j in range(self.imle_sample_factor)])
        )
        distances = jnp.sum(jnp.square(candidates - actions[None]), axis=(-2, -1))
        winner = jnp.argmin(distances, axis=0)
        winning_latent = jnp.take_along_axis(latents, winner[None, ..., None, None], axis=0)[0]

        # Update step (Eq. 4): only the winner is recomputed with gradients.
        predicted = self._generate(observation, winning_latent, kv_cache, prefix_mask)
        return jnp.mean(jnp.square(predicted - actions), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        del num_steps  # single-step by construction; kept so the caller's signature is unchanged
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        kv_cache, prefix_mask = self._prefix_cache(observation)
        return self._generate(observation, noise, kv_cache, prefix_mask)
