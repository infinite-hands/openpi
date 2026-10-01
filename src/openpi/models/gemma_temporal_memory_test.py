import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma


def _module():
    config = gemma.get_config("dummy")
    module = gemma.Module(configs=[config, config], embed_dtype="float32")
    batch, tokens = 2, 6
    embedded = [jax.random.normal(jax.random.key(1), (batch, tokens, config.width)), None]
    positions = jnp.broadcast_to(jnp.arange(tokens), (batch, tokens))
    mask = jnp.ones((batch, tokens, tokens), dtype=bool)
    params = nn.Module.init(module, jax.random.key(0), [False, False], method="init")  # every expert's params
    return config, module, params, embedded, positions, mask


def _memory(config, batch, slots, *, active, valid=True, seed=3):
    depth, heads, dim = config.depth, config.num_kv_heads, config.head_dim
    keys = jax.random.split(jax.random.key(seed), 2)
    return {
        "k": jax.random.normal(keys[0], (depth, batch, slots, heads, dim)),
        "v": jax.random.normal(keys[1], (depth, batch, slots, heads, dim)),
        "valid": jnp.full((depth, batch, slots), valid),
        "bias": jnp.zeros((depth, batch, heads, slots)),
        "active": jnp.asarray(active),
    }


def test_inactive_or_empty_memory_changes_nothing():
    config, module, params, embedded, positions, mask = _module()
    outputs, kv_cache = module.apply(params, embedded, positions, mask)
    for memory in (_memory(config, 2, 12, active=[False] * config.depth),
                   _memory(config, 2, 12, active=[True] * config.depth, valid=False)):
        with_memory, memory_kv_cache, memory_kv = module.apply(params, embedded, positions, mask,
                                                               temporal_memory=memory)
        np.testing.assert_array_equal(outputs[0], with_memory[0])
        np.testing.assert_array_equal(kv_cache[0], memory_kv_cache[0])
        np.testing.assert_array_equal(kv_cache[1], memory_kv_cache[1])
        assert memory_kv[0].shape == (config.depth, 2, 6, config.num_kv_heads, config.head_dim)


def test_an_active_layer_loads_history_and_keeps_norms():
    config, module, params, embedded, positions, mask = _module()
    active = [False, True] + [False] * (config.depth - 2)
    _, base_cache, base_raw = module.apply(params, embedded, positions, mask,
                                           temporal_memory=_memory(config, 2, 12, active=[False] * config.depth))
    outputs, cache, raw = module.apply(params, embedded, positions, mask,
                                       temporal_memory=_memory(config, 2, 12, active=active))
    np.testing.assert_array_equal(raw[0][0], base_raw[0][0])  # layer 0 runs before any loaded layer
    assert not np.allclose(cache[0][1], base_cache[0][1])      # layer 1's keys took the readout
    k = base_raw[0][1]
    memory = {name: value[1] for name, value in _memory(config, 2, 12, active=active).items()}
    loaded, _ = gemma._temporal_memory(k, base_raw[1][1], memory, config.head_dim)
    np.testing.assert_allclose(np.linalg.norm(loaded, axis=-1), np.linalg.norm(k, axis=-1), rtol=1e-5)
    assert np.isfinite(np.asarray(outputs[0])).all()


def test_a_suffix_pass_reads_the_loaded_cache():
    config, module, params, embedded, positions, mask = _module()
    _, cache, _raw = module.apply(params, embedded, positions, mask,
                                  temporal_memory=_memory(config, 2, 12, active=[True] * config.depth))
    suffix = [None, jax.random.normal(jax.random.key(4), (2, 3, config.width))]
    suffix_positions = jnp.broadcast_to(jnp.arange(6, 9), (2, 3))
    suffix_mask = jnp.ones((2, 3, 9), dtype=bool)
    (_, out), _ = module.apply(params, suffix, suffix_positions, suffix_mask, kv_cache=cache)
    assert out.shape == (2, 3, config.width) and np.isfinite(np.asarray(out)).all()
