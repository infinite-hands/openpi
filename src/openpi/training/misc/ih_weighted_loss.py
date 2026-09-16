"""pi0.5 with a critical-window weighted flow-matching loss and task-space auxiliary terms.

Vendored copy of infinite-hands/infinite-hands's models/vla/pi/weighted_loss.py, adapted to import
its dependencies from openpi.training.misc.ih_* instead of the main repo (this fork can't import
it). Keep in sync with that file by hand.

No parameters are added relative to plain Pi0, so any plain-Pi0 checkpoint warm-starts it, and
`sample_actions` is inherited, so serving is unchanged.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from flax import nnx
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize
from typing_extensions import override

from openpi.training.misc.ih_hardware_constants import STATE_DIM
from openpi.training.misc.ih_loss_math import (
    absolute_joints,
    critical_window,
    dim_weights,
    ee_error,
    timing_error,
    unnormalize_quantile,
)

ACTIONS_STATS_KEY = "actions"
STATE_STATS_KEY = "state"


@dataclasses.dataclass(frozen=True)
class WeightedLossConfig:
    """Every knob of the weighted loss; each field is a `--model.loss.*` trainer flag."""

    window_gain: float = 4.0          # extra weight at a gripper transition, on top of the base 1
    window_sigma_frames: float = 6.0  # +-0.2 s at 30 fps
    velocity_ref: float = 0.25        # normalized gripper units/frame that count as a full transition
    gripper_dim_gain: float = 4.0
    ee_weight: float = 20.0           # m^2 -> loss units: a 3 cm miss at a critical step ~ the flow loss
    ee_xy_weight: float = 2.0
    ee_z_weight: float = 3.0          # > ee_xy_weight: corrects a systematic shallow-grasp bias (insufficient descent), not symmetric noise
    timing_weight: float = 0.0
    timing_temperature: float = 0.1
    aux_time_power: float = 2.0       # aux terms gated by (1 - t)^p: the clean-action estimate is noise at t ~ 1
    norm_stats_dir: str = ""          # directory holding norm_stats.json; required when an aux term is on

    def needs_norm_stats(self) -> bool:
        return self.ee_weight > 0 or self.timing_weight > 0


@dataclasses.dataclass(frozen=True)
class Pi0WeightedConfig(pi0_config.Pi0Config):
    loss: WeightedLossConfig = WeightedLossConfig()

    @override
    def create(self, rng: at.KeyArrayLike) -> Pi0Weighted:
        return Pi0Weighted(self, rngs=nnx.Rngs(rng))


@functools.lru_cache(maxsize=4)
def _quantiles(norm_stats_dir: str) -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
    if not norm_stats_dir:
        raise ValueError("WeightedLossConfig.norm_stats_dir is required when ee_weight or timing_weight > 0")
    stats = _normalize.load(norm_stats_dir)
    out = {}
    for key in (ACTIONS_STATS_KEY, STATE_STATS_KEY):
        entry = stats[key]
        if entry.q01 is None or entry.q99 is None:
            raise ValueError(f"{norm_stats_dir}/norm_stats.json has no quantiles for {key!r}")
        out[key] = (tuple(float(v) for v in entry.q01[:STATE_DIM]), tuple(float(v) for v in entry.q99[:STATE_DIM]))
    return out


class Pi0Weighted(_pi0.Pi0):
    def __init__(self, config: Pi0WeightedConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.loss_cfg = config.loss
        quantiles = _quantiles(config.loss.norm_stats_dir) if config.loss.needs_norm_stats() else None
        self.actions_quantiles = quantiles[ACTIONS_STATS_KEY] if quantiles else None
        self.state_quantiles = quantiles[STATE_STATS_KEY] if quantiles else None

    def _flow(self, rng, observation, actions, *, train):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        return observation, v_t, u_t, x_t, time

    def compute_base_loss(self, rng, observation, actions, *, train=False):
        """openpi's unweighted objective, so a weighted and a plain checkpoint score on the same function."""
        _, v_t, u_t, _, _ = self._flow(rng, observation, actions, train=train)
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def compute_loss(self, rng, observation, actions, *, train=False):
        return self.compute_loss_terms(rng, observation, actions, train=train)[0]

    def compute_loss_terms(self, rng, observation, actions, *, train=False):
        """Per-(batch, step) weighted loss plus the scalar terms the trainer logs beside it."""
        cfg = self.loss_cfg
        observation, v_t, u_t, x_t, time = self._flow(rng, observation, actions, train=train)
        squared = jnp.square(v_t - u_t)

        critical, window = critical_window(actions, cfg.velocity_ref, cfg.window_gain, cfg.window_sigma_frames, jnp)
        window = jax.lax.stop_gradient(window)
        loss_flow = window * jnp.mean(squared * dim_weights(actions.shape[-1], cfg.gripper_dim_gain, jnp), axis=-1)
        chunked = loss_flow
        aux = {"loss_flow": jnp.mean(loss_flow), "window_mean": jnp.mean(critical)}

        if cfg.needs_norm_stats():
            gate = (1.0 - time) ** cfg.aux_time_power
            a_hat = x_t - time[..., None, None] * v_t
            state = unnormalize_quantile(observation.state, *self.state_quantiles, jnp)
            predicted = absolute_joints(unnormalize_quantile(a_hat, *self.actions_quantiles, jnp), state, jnp)
            expected = jax.lax.stop_gradient(
                absolute_joints(unnormalize_quantile(actions, *self.actions_quantiles, jnp), state, jnp))
            if cfg.ee_weight > 0:
                loss_ee = gate[..., None] * window * ee_error(predicted, expected, cfg.ee_xy_weight, cfg.ee_z_weight, jnp)
                chunked = chunked + cfg.ee_weight * loss_ee
                aux["loss_ee"] = jnp.mean(loss_ee)
            if cfg.timing_weight > 0:
                loss_timing = gate * timing_error(a_hat, actions, cfg.velocity_ref, cfg.timing_temperature, jnp)
                chunked = chunked + cfg.timing_weight * loss_timing[..., None]
                aux["loss_timing"] = jnp.mean(loss_timing)
        return chunked, aux
