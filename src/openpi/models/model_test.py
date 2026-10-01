from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


def test_load_zero_missing_lora():
    config = pi0_config.Pi0Config(paligemma_variant="gemma_300m_lora", action_expert_variant="gemma_300m_lora")
    _, state = nnx.split(config.create(jax.random.key(0)))
    flat = traverse_util.flatten_dict(state.to_pure_dict(), sep="/")
    lora_paths = {path for path in flat if "lora" in path}
    assert lora_paths
    without_lora = {path: leaf for path, leaf in flat.items() if path not in lora_paths}
    dtype = next(iter(without_lora.values())).dtype

    with pytest.raises(ValueError):
        config.load(traverse_util.unflatten_dict(without_lora, sep="/"))

    model = config.load(traverse_util.unflatten_dict(without_lora, sep="/"), zero_missing_regex=".*lora.*")
    _, loaded = nnx.split(model)
    loaded_flat = traverse_util.flatten_dict(loaded.to_pure_dict(), sep="/")
    assert set(loaded_flat) == set(flat)
    for path in lora_paths:
        assert loaded_flat[path].shape == flat[path].shape, path
        assert loaded_flat[path].dtype == dtype, path
        assert not jnp.any(loaded_flat[path]), path

    non_lora = next(iter(without_lora))
    also_missing = {path: leaf for path, leaf in without_lora.items() if path != non_lora}
    with pytest.raises(ValueError):
        config.load(traverse_util.unflatten_dict(also_missing, sep="/"), zero_missing_regex=".*lora.*")


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
