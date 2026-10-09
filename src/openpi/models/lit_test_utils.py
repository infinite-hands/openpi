"""Shared CPU harness for the LIT (Latent Interface Training) tests: a tiny openpi Pi0 and deterministic inputs.

The public API below is stable; later LIT tests import it unchanged.

    make_tiny_config(**overrides) -> Pi0Config
    make_observation(seed, *, config=None, batch=BATCH, masked=..., **fields) -> Observation
    make_actions(seed, *, config=None, batch=BATCH) -> Actions
    build_model(config, seed=0) -> Pi0
    randomize_zero_init(model, seed=0, scale=0.02) -> Pi0
    fixed_noise(seed, *, config=None, batch=BATCH)                 noise for sample_actions(noise=...)
    training_noise_and_time(rng, batch, config=None)               what compute_loss draws from `rng`
    jit_loss(model) / jit_sample(model)                            module_jit wrappers, as Policy uses them
    param_manifest(model) / array_digest(x)                        path -> shape, dtype, sha256 of the values
    stub_image_encoder_enabled()                                   True when the opt-in stub encoder is active
    stub_image_encoder()                                           context manager: the stub encoder inside it

What the tiny model is. Both Gemma experts use the "dummy" variant (width 64, depth 4, 1 kv head of head_dim 16),
pi05=True, float32, action_dim 14, action_horizon 4, max_token_len 8. The image tower is the REAL SigLIP So400m/14 by
default: Pi0.__init__ hard-codes it and the config offers no smaller variant. Images must be 224x224 (other sizes are
resized inside preprocess_observation), giving 256 image tokens per camera; with 3 cameras and max_token_len 8 the
prefix is 3 * 256 + 8 = 776 tokens (openpi has no history blocks and no memory: one prefix, [cameras | language]).

Opt-in stub image encoder (speed only). With LIT_TEST_STUB_IMAGE_ENCODER=1 in the environment, build_model replaces
the SigLIP tower by a single zero-initialised Dense over the 14x14 patches of the image (same (b, 256, width) output,
same `PaliGemma/img/head/kernel` path, nothing else). It is test code, off by default, and it never changes what a
default build computes. A stub model draws different random numbers than a real-SigLIP one, so it cannot reproduce the
lit_golden fixtures: every test that compares against them skips (see gen_baseline.fixture_unavailable_reason) when the
stub is on. Run the final LIT test suites without the env var.

Two init facts the tests depend on. A freshly created pi0.5 is prefix-blind: every adaRMS modulation kernel is
zero-initialised, so the action expert's attention contributes nothing. And the SigLIP head (head_zeroinit) is
zero-initialised, so every image token is exactly zero. Call randomize_zero_init(model, seed) to get a model that
actually reads the prefix and the images; tests of masking, gradients or image/language dependence need it.

LIT fields. `lit` and `lit_*` overrides are forwarded to Pi0Config only when the field exists, so this file works
before and after the LIT fields land. On a Pi0Config without them, `lit="off"` (or no lit override) is accepted and
ignored, and any other lit request raises ValueError instead of silently building a stock model. With lit != "off"
the tiny defaults in TINY_LIT_DEFAULTS fill every lit_* field the config has and the caller did not set
(lit_groups=2 divides the dummy depth 4; lit_kv_dim=16 equals the dummy head_dim, as 256 does for the real model).
make_observation passes extra Observation fields (for example the LIT goal) through **fields and rejects names the
dataclass does not have.
"""

import contextlib
import dataclasses
import hashlib
import os
from unittest import mock
import zlib

import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0_config as _pi0_config
from openpi.models import siglip as _siglip
from openpi.shared import nnx_utils

ACTION_DIM = 14
ACTION_HORIZON = 4
MAX_TOKEN_LEN = 8
BATCH = 2
# Dimensions of the state the fake robot actually drives; the rest of state and actions is zero padding.
ROBOT_DIMS = 7
# One camera of the second sample is missing (zero image, mask False), as for an unplugged or masked camera.
DEFAULT_MASKED = (("right_wrist_0_rgb", 1),)

STUB_ENV_VAR = "LIT_TEST_STUB_IMAGE_ENCODER"

TINY_LIT_DEFAULTS = {
    "lit_num_latents": 6,
    "lit_dim": 32,
    "lit_groups": 2,
    "lit_heads": 4,
    "lit_kv_dim": 16,
    "lit_pose_tokens": 2,
    "lit_goal_tokens": 2,
    "lit_goal_dims": tuple(range(ROBOT_DIMS)),
}

_CONFIG_DEFAULTS = {
    "paligemma_variant": "dummy",
    "action_expert_variant": "dummy",
    "pi05": True,
    "dtype": "float32",
    "action_dim": ACTION_DIM,
    "action_horizon": ACTION_HORIZON,
    "max_token_len": MAX_TOKEN_LEN,
}


def _is_lit(name: str) -> bool:
    return name == "lit" or name.startswith("lit_")


def make_tiny_config(**overrides) -> _pi0_config.Pi0Config:
    """The tiny pi0.5 config. `overrides` replace any default (and any Pi0Config field); see the module docstring for
    how lit / lit_* overrides are handled."""
    fields = {f.name for f in dataclasses.fields(_pi0_config.Pi0Config)}
    lit_overrides = {k: overrides.pop(k) for k in list(overrides) if _is_lit(k)}
    lit_mode = lit_overrides.get("lit", "off")
    lit_kwargs = dict(lit_overrides)
    if lit_mode != "off":
        lit_kwargs = {**{k: v for k, v in TINY_LIT_DEFAULTS.items() if k in fields}, **lit_overrides}
    missing = sorted(k for k in lit_kwargs if k not in fields)
    if missing:
        if lit_mode != "off":
            raise ValueError(f"Pi0Config has no field(s) {missing}: cannot build lit={lit_mode!r} yet")
        lit_kwargs = {k: v for k, v in lit_kwargs.items() if k in fields}
    return _pi0_config.Pi0Config(**{**_CONFIG_DEFAULTS, **overrides, **lit_kwargs})


def stub_image_encoder_enabled() -> bool:
    return os.environ.get(STUB_ENV_VAR, "") not in ("", "0")


class _StubSiglip(nn.Module):
    """One zero-initialised Dense over 14x14 patches: the opt-in stand-in for SigLIP (see the module docstring)."""

    num_classes: int
    variant: str = ""
    pool_type: str = ""
    scan: bool = False
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, image, *, train=False):
        b, h, w, c = image.shape
        p = 14
        patches = image.reshape(b, h // p, p, w // p, p, c).transpose(0, 1, 3, 2, 4, 5).reshape(b, -1, p * p * c)
        tokens = nn.Dense(self.num_classes, kernel_init=nn.initializers.zeros, dtype=self.dtype_mm, name="head")(
            patches
        )
        return tokens, {}


def stub_image_encoder():
    """Context manager: every Pi0 created inside it (build_model, a trainer's init_train_state) gets the stub image
    tower instead of SigLIP. For tests whose subject is not the image tower (the trainer, the loaders): the tree and
    the paths differ from a real-SigLIP model's only under `PaliGemma/img`."""
    return mock.patch.object(_siglip, "Module", _StubSiglip)


def build_model(config: _pi0_config.Pi0Config, seed: int = 0):
    with stub_image_encoder() if stub_image_encoder_enabled() else contextlib.nullcontext():
        return config.create(jax.random.key(seed))


def _flat_state(model: nnx.Module) -> dict:
    return nnx.state(model).flat_state()


def _path_str(path) -> str:
    return "/".join(str(p) for p in path)


def randomize_zero_init(model: nnx.Module, seed: int = 0, scale: float = 0.02):
    """A copy of `model` where every all-zero floating-point leaf gets scale * N(0, 1) noise.

    Each leaf's key is fold_in(key(seed), crc32(path)), so the values of existing leaves do not change when other
    parameters are added to the model. Non-zero leaves are untouched."""
    graphdef, state = nnx.split(model)
    base = jax.random.key(seed)
    flat = {}
    for path, var in state.flat_state().items():
        x = var.value
        if jnp.issubdtype(x.dtype, jnp.floating) and not bool(jnp.any(x)):
            key = jax.random.fold_in(base, zlib.crc32(_path_str(path).encode()))
            var = var.replace(value=x + scale * jax.random.normal(key, x.shape, x.dtype))  # noqa: PLW2901
        flat[path] = var
    return nnx.merge(graphdef, nnx.State.from_flat_path(flat))


def make_observation(
    seed: int,
    *,
    config: _pi0_config.Pi0Config | None = None,
    batch: int = BATCH,
    masked: tuple[tuple[str, int], ...] = DEFAULT_MASKED,
    **fields,
) -> _model.Observation:
    """A deterministic observation: uniform [-1, 1] images (b, 224, 224, 3) for the three cameras, a state that is
    nonzero on the first ROBOT_DIMS dims, and prompts of different lengths (max_token_len - 3 * i for sample i, at
    least 2) padded with token 0 and mask False. `masked` lists (camera, sample) pairs that are missing: zero image,
    mask False. `fields` replace Observation fields (e.g. the LIT goal) after construction."""
    config = config or make_tiny_config()
    rng = np.random.default_rng(seed)
    images, image_masks = {}, {}
    for name in _model.IMAGE_KEYS:
        image = rng.uniform(-1.0, 1.0, (batch, *_model.IMAGE_RESOLUTION, 3)).astype(np.float32)
        valid = np.ones((batch,), dtype=bool)
        for camera, sample in masked:
            if camera == name and sample < batch:
                image[sample] = 0.0
                valid[sample] = False
        images[name], image_masks[name] = jnp.asarray(image), jnp.asarray(valid)
    state = np.zeros((batch, config.action_dim), dtype=np.float32)
    state[:, :ROBOT_DIMS] = rng.normal(size=(batch, ROBOT_DIMS))
    lengths = np.maximum(2, config.max_token_len - 3 * np.arange(batch))
    prompt_mask = np.arange(config.max_token_len)[None, :] < lengths[:, None]
    prompt = np.where(prompt_mask, rng.integers(2, 250, (batch, config.max_token_len)), 0).astype(np.int32)
    observation = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=jnp.asarray(state),
        tokenized_prompt=jnp.asarray(prompt),
        tokenized_prompt_mask=jnp.asarray(prompt_mask),
    )
    if fields:
        known = {f.name for f in dataclasses.fields(_model.Observation)}
        if unknown := sorted(set(fields) - known):
            raise ValueError(f"Observation has no field(s) {unknown}")
        observation = observation.replace(**fields)
    return observation


def make_actions(seed: int, *, config: _pi0_config.Pi0Config | None = None, batch: int = BATCH) -> jax.Array:
    config = config or make_tiny_config()
    actions = np.zeros((batch, config.action_horizon, config.action_dim), dtype=np.float32)
    actions[..., :ROBOT_DIMS] = 0.5 * np.random.default_rng(seed).normal(size=(*actions.shape[:2], ROBOT_DIMS))
    return jnp.asarray(actions)


def fixed_noise(seed: int, *, config: _pi0_config.Pi0Config | None = None, batch: int = BATCH) -> jax.Array:
    config = config or make_tiny_config()
    return jax.random.normal(jax.random.key(seed), (batch, config.action_horizon, config.action_dim))


def training_noise_and_time(rng, batch: int = BATCH, config: _pi0_config.Pi0Config | None = None):
    """(noise, time): what Pi0.compute_loss draws from `rng` (3-way split: preprocess, noise, time). Mirrors the stock
    code, so a LIT test can rebuild x_t = time * noise + (1 - time) * actions with time[..., None, None]."""
    config = config or make_tiny_config()
    _, noise_rng, time_rng = jax.random.split(rng, 3)
    noise = jax.random.normal(noise_rng, (batch, config.action_horizon, config.action_dim))
    time = jax.random.beta(time_rng, 1.5, 1, (batch,)) * 0.999 + 0.001
    return noise, time


def jit_loss(model):
    return nnx_utils.module_jit(model.compute_loss, static_argnames=("train",))


def jit_sample(model):
    return nnx_utils.module_jit(model.sample_actions)


def array_digest(x) -> dict:
    """Shape, dtype, float64 sum and sha256 of the raw bytes: an exact-equality fingerprint of an array."""
    a = np.ascontiguousarray(np.asarray(x))
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "sum": float(np.sum(a.astype(np.float64))),
        "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
    }


def param_manifest(model: nnx.Module) -> dict:
    """{path: {type, shape, dtype, sum, sha256}} over every variable of the model, sorted by path."""
    return {
        _path_str(path): {"type": var.type.__name__, **array_digest(var.value)}
        for path, var in sorted(_flat_state(model).items(), key=lambda kv: _path_str(kv[0]))
    }
