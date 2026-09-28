"""WAM future tokens (Pi0.compute_loss_with_future): the invariants the mechanism study relies on.

The key claim is that train_only mode leaves the action pathway untouched: same keys, same
positions, no dependence on the future tokens, so inference is unchanged and any effect of the
future loss arrives only through shared weights.
"""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
import openpi.models.pi0_config as _pi0_config
from openpi.shared import nnx_utils
from openpi.training import wam_aux
from openpi.training import weight_loaders

BATCH = 2
_STOCK = _pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", dtype="float32")
_TRAIN_ONLY = dataclasses.replace(_STOCK, future_tokens="train_only", future_head_hidden=16)
_VISIBLE = dataclasses.replace(_STOCK, future_tokens="visible", future_head_hidden=16)
_FUTURE = nnx_utils.PathRegex("future_.*")


def _observation_and_actions(config: _pi0_config.Pi0Config) -> tuple[_model.Observation, _model.Actions]:
    """Random inputs with different prompt lengths per example and one missing camera."""
    keys = jax.random.split(jax.random.key(7), 6)
    image_shape = (BATCH, *_model.IMAGE_RESOLUTION, 3)
    images = {
        name: jax.random.uniform(key, image_shape, minval=-1.0, maxval=1.0)
        for name, key in zip(_model.IMAGE_KEYS, keys[:3], strict=True)
    }
    image_masks = {name: jnp.ones((BATCH,), dtype=jnp.bool_) for name in _model.IMAGE_KEYS}
    image_masks["left_wrist_0_rgb"] = jnp.array([True, False])
    lengths = jnp.array([config.max_token_len // 3, config.max_token_len // 2])
    prompt_mask = jnp.arange(config.max_token_len)[None] < lengths[:, None]
    observation = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=jax.random.normal(keys[3], (BATCH, config.action_dim)),
        tokenized_prompt=jax.random.randint(keys[4], (BATCH, config.max_token_len), 0, 256),
        tokenized_prompt_mask=prompt_mask,
    )
    actions = jax.random.normal(keys[5], (BATCH, config.action_horizon, config.action_dim))
    return observation, actions


def _without_zero_params(model: nnx.Module) -> nnx.Module:
    """The model with every all-zero float parameter given small random values.

    A freshly initialised pi05 action expert starts as an identity: its adaRMS modulation (scale,
    shift and residual gate) is zero-initialised, so no suffix block mixes tokens and every
    attention path through the suffix -- the ones these tests are about -- carries nothing. A
    trained checkpoint has nonzero gates; this makes a fresh model behave like one.
    """
    graphdef, state = nnx.split(model)
    leaves, treedef = jax.tree_util.tree_flatten(state)
    keys = jax.random.split(jax.random.key(11), len(leaves))
    leaves = [
        x + 0.02 * jax.random.normal(key, x.shape, x.dtype)
        if jnp.issubdtype(x.dtype, jnp.floating) and not jnp.any(x)
        else x
        for x, key in zip(leaves, keys, strict=True)
    ]
    return nnx.merge(graphdef, jax.tree_util.tree_unflatten(treedef, leaves))


@pytest.fixture(scope="module")
def models():
    """A train_only model, a visible model and a stock model sharing every stock parameter."""
    train_only = _without_zero_params(_TRAIN_ONLY.create(jax.random.key(0)))
    visible = _VISIBLE.load(nnx.state(train_only).to_pure_dict(), remove_extra_params=True)
    stock = _STOCK.load(nnx.state(train_only).to_pure_dict(), remove_extra_params=True)
    return train_only, visible, stock


def _stock_mask_and_positions(prefix_mask, ar_mask):
    valid = jnp.concatenate([prefix_mask, jnp.ones((prefix_mask.shape[0], len(ar_mask) - prefix_mask.shape[1]), bool)], 1)
    return _pi0.make_attn_mask(valid, jnp.array(ar_mask)), jnp.cumsum(valid, axis=1) - 1


def test_mask_and_positions():
    n_obs, n_fp, n_fs, n_act = 7, 3, 4, 5
    prefix_mask = jnp.array([[True] * 5 + [False] * 2, [True] * 7])

    # No future tokens: exactly the stock pi05 mask and positions.
    mask, positions = _pi0.make_future_attn_mask_and_positions(prefix_mask, 0, 0, n_act, visible=False)
    stock_mask, stock_positions = _stock_mask_and_positions(prefix_mask, [False] * n_obs + [True] + [False] * (n_act - 1))
    np.testing.assert_array_equal(mask, stock_mask)
    np.testing.assert_array_equal(positions, stock_positions)

    # train_only: the obs and act rows/columns equal the stock model's, with act at stock positions.
    mask, positions = _pi0.make_future_attn_mask_and_positions(prefix_mask, n_fp, n_fs, n_act, visible=False)
    keep = np.r_[0:n_obs, n_obs + n_fp + n_fs : n_obs + n_fp + n_fs + n_act]
    np.testing.assert_array_equal(mask[:, keep][:, :, keep], stock_mask)
    np.testing.assert_array_equal(positions[:, keep], stock_positions)
    n_valid = prefix_mask.sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(positions[:, n_obs : n_obs + n_fp], n_valid + jnp.arange(n_fp))
    # Nothing but Fs itself reads Fs, and nothing but Fp itself reads Fp.
    assert not mask[:, :n_obs, n_obs:].any()
    assert not mask[:, n_obs + n_fp + n_fs :, n_obs : n_obs + n_fp + n_fs].any()

    # visible: exactly make_attn_mask with the block starts, and cumsum positions.
    mask, positions = _pi0.make_future_attn_mask_and_positions(prefix_mask, n_fp, n_fs, n_act, visible=True)
    ar = [False] * n_obs + [True] + [False] * (n_fp - 1) + [True] + [False] * (n_fs + n_act - 1)
    visible_mask, visible_positions = _stock_mask_and_positions(prefix_mask, ar)
    np.testing.assert_array_equal(mask, visible_mask)
    np.testing.assert_array_equal(positions, visible_positions)


@pytest.mark.parametrize("train", [False, True])
def test_train_only_matches_stock_action_loss(models, train):
    train_only, _, stock = models
    observation, actions = _observation_and_actions(_STOCK)
    rng = jax.random.key(3)
    action_loss = train_only.compute_loss_with_future(rng, observation, actions, train=train)[0]
    # Allclose, not bitwise: the softmax now also sums the masked-out future keys (as exact zeros),
    # which reorders the float adds.
    np.testing.assert_allclose(action_loss, stock.compute_loss(rng, observation, actions, train=train), rtol=1e-5, atol=1e-5)


def test_train_only_action_loss_ignores_future_params(models):
    train_only, _, _ = models
    observation, actions = _observation_and_actions(_STOCK)
    rng = jax.random.key(3)
    before = train_only.compute_loss_with_future(rng, observation, actions)[0]
    graphdef, state = nnx.split(train_only)
    perturbed_state = jax.tree_util.tree_map_with_path(
        lambda path, x: x * 3.0 + 1.0 if "future_" in jax.tree_util.keystr(path) else x, state
    )
    after = nnx.merge(graphdef, perturbed_state).compute_loss_with_future(rng, observation, actions)[0]
    np.testing.assert_array_equal(before, after)


def test_train_only_gradient_isolation(models):
    train_only, _, _ = models
    observation, actions = _observation_and_actions(_STOCK)
    rng = jax.random.key(3)

    def action_loss(model):
        return jnp.mean(model.compute_loss_with_future(rng, observation, actions)[0])

    def future_loss(model):
        _, prefix, suffix, _ = model.compute_loss_with_future(rng, observation, actions)
        return jnp.mean(prefix) + jnp.mean(suffix)

    action_grads = nnx.grad(action_loss)(train_only)
    assert all(jnp.all(g == 0) for g in jax.tree.leaves(action_grads.filter(_FUTURE)))

    future_grads = nnx.grad(future_loss)(train_only)
    for pattern in (".*img.*", ".*llm.*", "action_in_proj.*", "future_.*"):
        leaves = jax.tree.leaves(future_grads.filter(nnx_utils.PathRegex(pattern)))
        assert leaves, pattern
        assert any(jnp.any(g != 0) for g in leaves), f"no future-loss gradient reaches {pattern}"


def test_train_only_sample_actions_unchanged(models):
    train_only, _, stock = models
    observation, _ = _observation_and_actions(_STOCK)
    noise = jax.random.normal(jax.random.key(5), (BATCH, _STOCK.action_horizon, _STOCK.action_dim))
    rng = jax.random.key(0)
    np.testing.assert_array_equal(
        train_only.sample_actions(rng, observation, noise=noise), stock.sample_actions(rng, observation, noise=noise)
    )


def test_visible_changes_actions_and_refuses_inference(models):
    _, visible, stock = models
    observation, actions = _observation_and_actions(_STOCK)
    rng = jax.random.key(3)
    action_loss = visible.compute_loss_with_future(rng, observation, actions)[0]
    assert not np.allclose(action_loss, stock.compute_loss(rng, observation, actions), atol=1e-6)

    grads = nnx.grad(lambda m: jnp.mean(m.compute_loss_with_future(rng, observation, actions)[0]))(visible)
    assert any(jnp.any(g != 0) for g in jax.tree.leaves(grads.filter(_FUTURE)))

    with pytest.raises(NotImplementedError):
        visible.sample_actions(rng, observation)
    with pytest.raises(NotImplementedError):
        visible.compute_loss(rng, observation, actions)


def test_future_patch_loss():
    keys = jax.random.split(jax.random.key(1), 3)
    target = jax.random.normal(keys[0], (3, 16, 8))
    result = wam_aux.future_patch_loss(target, target, target)
    np.testing.assert_allclose(result["loss"], 0.0, atol=1e-5)
    np.testing.assert_allclose(result["copy"], 0.0, atol=1e-5)

    # A padded example is excluded from the mean, however wrong its prediction.
    wrong = target.at[2].set(jax.random.normal(keys[1], (16, 8)))
    padded = wam_aux.future_patch_loss(wrong, target, target, is_pad=jnp.array([False, False, True]))
    unpadded = wam_aux.future_patch_loss(wrong, target, target)
    assert float(unpadded["loss"]) > 0.05
    # Centering uses the batch mean, so the padded example still shifts the valid ones slightly.
    assert float(padded["loss"]) < float(unpadded["loss"]) / 3

    with pytest.raises(ValueError):
        wam_aux.future_patch_loss(target[:, :8], target, target)


def test_loader_leaves_future_params_fresh():
    reference = {
        "action_in_proj": {"kernel": jax.ShapeDtypeStruct((2, 2), jnp.float32)},
        "future_prefix_shared": jax.ShapeDtypeStruct((1, 1, 4), jnp.float32),
    }
    loaded = {"action_in_proj": {"kernel": np.ones((2, 2), np.float32)}}
    merged = weight_loaders._merge_params(loaded, reference, missing_regex=".*lora.*|future_.*")  # noqa: SLF001
    assert isinstance(merged["future_prefix_shared"], jax.ShapeDtypeStruct)
    stock = weight_loaders._merge_params(loaded, reference, missing_regex=".*lora.*")  # noqa: SLF001
    assert "future_prefix_shared" not in stock
