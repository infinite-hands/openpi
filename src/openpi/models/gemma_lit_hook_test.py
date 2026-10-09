"""The opt-in per-layer input hook of the scanned Gemma (Module/Block `return_layer_inputs`), the static_argnums that
carry it, and the pre-extended cache LIT uses to append latent K/V columns.

Everything runs on the "dummy" Gemma variant (width 64, depth 4, 1 kv head of head_dim 16) with two experts, the second
one adaRMS-conditioned as in pi0.5, so it needs no SigLIP and takes seconds. The inputs and parameters are those of
lit_golden/gen_baseline (the identity fixture's), so the flagged calls can be compared with the unflagged ones and with
the fixture made from the unmodified pin. Reference values come from running the layers one at a time (Block.apply on
one layer's slice of the stacked params), from truncated-depth Modules, and from a numpy float64 attention; none of them
goes through the capture code.

Float comparisons use _REL, a measured scan-versus-loop float32 difference with headroom.
"""

import dataclasses
import inspect
from unittest import mock

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models.lit_golden import gen_baseline as _gen

INPUTS = _gen.gemma_inputs()
CONFIGS = INPUTS["configs"]
B = _gen.GEMMA_BATCH
PREFIX = _gen.GEMMA_PREFIX
SUFFIX = _gen.GEMMA_SUFFIX
DEPTH = CONFIGS[0].depth
WIDTH = CONFIGS[0].width
HEAD_DIM = CONFIGS[0].head_dim
LATENTS = 5
# Largest difference to the reference, as a fraction of that array's largest magnitude, measured on this CPU (float32):
# see _MEASURED below. The bound leaves more than 10x headroom over every one of them.
_REL = 1e-5
# bfloat16 rounds to 8 bits: one rounding step is 2**-8 = 3.9e-3 of the value.
_BF16_REL = 2**-8

# The call paths of the scanned Gemma in which the first expert runs (the others have no layer input to return).
_FIRST_EXPERT_PATHS = (
    "joint",
    "joint_bf16",
    "prefix_only",
    "collect_attention",
    "collect_attention_selected",
    "pool_prefix_expert",
    "feature_band",
    "feature_shift_token_mask",
    "attention_pool_feature",
    "dropout_train",
    "dropout_eval",
)
_ACTION_EXPERT_PATHS = ("suffix_over_cache", "pool_action_expert", "feature_action_expert")


@pytest.fixture(scope="module")
def case():
    variables = _gen.gemma_variables(INPUTS)
    return {"module": _gen.gemma_module(INPUTS), "variables": variables, **INPUTS}


def _joint(c, **kwargs):
    return c["module"].apply(
        c["variables"],
        [c["prefix"], c["suffix"]],
        c["positions"],
        c["mask"],
        [None, c["cond"]],
        **kwargs,
    )


def _prefix_only(c, **kwargs):
    return c["module"].apply(c["variables"], [c["prefix"], None], c["prefix_positions"], c["prefix_mask"], **kwargs)


def _layer_params(variables, i):
    return jax.tree.map(lambda p: p[i], variables["params"]["layers"])


def _layer_by_layer(variables, embedded, positions, mask, adarms_cond):
    """The first expert's input to every layer, from running the Blocks one at a time; plus the final residuals."""
    block = gemma.Block(configs=tuple(CONFIGS))
    xs, inputs = list(embedded), []
    for i in range(DEPTH):
        inputs.append(xs[0])
        xs, _ = block.apply(
            {"params": _layer_params(variables, i)}, xs, None, positions, mask[:, None], adarms_cond, deterministic=True
        )
    return jnp.stack(inputs), xs


def _close(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    assert got.shape == want.shape
    assert np.max(np.abs(got - want)) <= _REL * np.max(np.abs(want)), np.max(np.abs(got - want))


def _equal(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        assert x.dtype == y.dtype
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


# ---- the default call is untouched ----


def test_default_call_keeps_its_return_structure(case):
    result = _joint(case)
    assert isinstance(result, tuple)
    assert len(result) == 2
    outputs, kv_cache = result
    assert [o.shape for o in outputs] == [(B, PREFIX, WIDTH), (B, SUFFIX, WIDTH)]
    assert [a.shape for a in kv_cache] == [(DEPTH, B, PREFIX + SUFFIX, 1, HEAD_DIM)] * 2


@pytest.mark.parametrize("name", _FIRST_EXPERT_PATHS)
def test_flag_changes_nothing_but_the_extra_last_return_on_every_call_path(name):
    """Every pre-existing call path (default, prefix-only, collect_attention, pooling, feature steering, dropout,
    bfloat16) with the flag on returns what it returned before, and the layer inputs after all of it. The extra scan
    output can change how XLA fuses the layer (the feature-steering path then differs from the plain call by float
    rounding, measured 1.2e-6 absolute on outputs of order 3 in float32), so values are compared within _REL; the flag
    OFF is exactly the pin (lit_identity_test.test_gemma_call_paths_are_bit_identical)."""
    plain = jax.tree.leaves(_gen.gemma_call_paths()[name]())
    flagged = _gen.gemma_call_paths(return_layer_inputs=True)[name]()
    assert len(jax.tree.leaves(flagged)) == len(plain) + 1
    *rest, layer_inputs = flagged
    assert layer_inputs.shape == (DEPTH, B, PREFIX, WIDTH)
    assert layer_inputs.dtype == (jnp.bfloat16 if name == "joint_bf16" else jnp.float32)
    rel = 0.0
    for got, want in zip(jax.tree.leaves(rest), plain, strict=True):
        assert got.dtype == want.dtype
        got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)  # noqa: PLW2901
        rel = max(rel, float(np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-30)))
    print(f"{name}: flagged vs plain, worst relative difference {rel:.2e}")
    tolerance = _BF16_REL if name == "joint_bf16" else _REL
    assert rel <= tolerance


@pytest.mark.parametrize("name", _ACTION_EXPERT_PATHS)
def test_the_flag_needs_the_first_expert_on_every_action_expert_path(name):
    with pytest.raises(ValueError, match="first expert"):
        _gen.gemma_call_paths(return_layer_inputs=True)[name]()


def test_result_order_with_attention_pooling_and_layer_inputs(case):
    outputs, kv_cache, attention, pooled, layer_inputs = _prefix_only(
        case,
        collect_attention=True,
        pool_mask=case["prefix_pool"],
        return_layer_inputs=True,
    )
    assert attention.shape == (DEPTH, B, PREFIX)  # mean over heads and queries: one weight per key
    assert pooled.shape == (DEPTH, B, WIDTH)
    assert layer_inputs.shape == (DEPTH, B, PREFIX, WIDTH)
    plain = _prefix_only(case, collect_attention=True, pool_mask=case["prefix_pool"])
    _equal(plain, (outputs, kv_cache, attention, pooled))


# ---- what the layer inputs are ----


def test_layer_inputs_match_running_the_layers_one_at_a_time(case):
    _, _, layer_inputs = _joint(case, return_layer_inputs=True)
    expected, _ = _layer_by_layer(
        case["variables"], [case["prefix"], case["suffix"]], case["positions"], case["mask"], [None, case["cond"]]
    )
    np.testing.assert_array_equal(np.asarray(layer_inputs[0]), np.asarray(case["prefix"]))
    _close(np.asarray(layer_inputs), np.asarray(expected))
    # the layers do change the stream: a wrong layer index would not be caught by the comparison above otherwise
    assert all(float(jnp.max(jnp.abs(layer_inputs[i + 1] - layer_inputs[i]))) > 1e-3 for i in range(DEPTH - 1))
    error = np.max(np.abs(np.asarray(layer_inputs, np.float64) - np.asarray(expected, np.float64)))
    print(
        f"layer inputs vs layer-by-layer: max abs difference {error:.2e}, scale {float(jnp.max(jnp.abs(expected))):.2e}"
    )


def test_prefix_only_pass_without_adarms(case):
    outputs, kv_cache, layer_inputs = _prefix_only(case, return_layer_inputs=True)
    expected, _ = _layer_by_layer(
        case["variables"], [case["prefix"], None], case["prefix_positions"], case["prefix_mask"], [None, None]
    )
    _close(np.asarray(layer_inputs), np.asarray(expected))
    _equal((outputs, kv_cache), _prefix_only(case))


def test_layer_inputs_match_truncated_depth_modules(case):
    """Through the scan path only: a Module of depth d, with the first d layers' params, ends in the final norm of the
    residual that enters layer d."""
    _, _, layer_inputs = _joint(case, return_layer_inputs=True)
    params = case["variables"]["params"]
    for d in range(1, DEPTH):
        configs = [dataclasses.replace(config, depth=d) for config in CONFIGS]
        truncated = gemma.Module(configs=configs, embed_dtype="float32", adarms=True)
        variables = {
            "params": {
                **{k: v for k, v in params.items() if k != "layers"},
                "layers": jax.tree.map(lambda p, d=d: p[:d], params["layers"]),
            }
        }
        outputs, _ = truncated.apply(
            variables,
            [case["prefix"], case["suffix"]],
            case["positions"],
            case["mask"],
            [None, case["cond"]],
        )
        norm = gemma.RMSNorm().apply({"params": params["final_norm"]}, layer_inputs[d], None)[0]
        _close(np.asarray(outputs[0]), np.asarray(norm))


@pytest.mark.parametrize("window", [(0, PREFIX), (2, 3), (PREFIX - 1, 1), (0, 1)])
def test_slice_returns_exactly_that_slice(case, window):
    start, length = window
    *_, everything = _joint(case, return_layer_inputs=True)
    outputs, kv_cache, sliced = _joint(case, return_layer_inputs=True, layer_input_slice=window)
    assert sliced.shape == (DEPTH, B, length, WIDTH)
    np.testing.assert_array_equal(np.asarray(sliced), np.asarray(everything[:, :, start : start + length]))
    _equal((outputs, kv_cache), _joint(case))


@pytest.mark.parametrize("window", [(-1, 2), (0, 0), (0, PREFIX + 1), (PREFIX, 1), (3, PREFIX)])
def test_slice_outside_the_tokens_is_refused(case, window):
    with pytest.raises(ValueError, match="layer_input_slice"):
        _joint(case, return_layer_inputs=True, layer_input_slice=window)


def test_the_first_expert_must_run(case):
    _, prefix_cache = _prefix_only(case)
    with pytest.raises(ValueError, match="first expert"):
        case["module"].apply(
            case["variables"],
            [None, case["suffix"]],
            case["suffix_positions"],
            case["suffix_mask"],
            [None, case["cond"]],
            kv_cache=prefix_cache,
            return_layer_inputs=True,
        )


def test_jit_matches_eager_and_is_deterministic(case):
    window = (1, 4)

    @jax.jit
    def run(variables, prefix, suffix, cond):
        return case["module"].apply(
            variables,
            [prefix, suffix],
            case["positions"],
            case["mask"],
            [None, cond],
            return_layer_inputs=True,
            layer_input_slice=window,
        )

    jitted = run(case["variables"], case["prefix"], case["suffix"], case["cond"])
    eager = _joint(case, return_layer_inputs=True, layer_input_slice=window)
    _close(np.asarray(jitted[2]), np.asarray(eager[2]))
    _close(np.asarray(jitted[0][1]), np.asarray(eager[0][1]))
    _equal(jitted, run(case["variables"], case["prefix"], case["suffix"], case["cond"]))


# ---- gradients through the layer inputs, with the remat in the program ----


def _weights(shape):
    return jax.random.normal(jax.random.key(7), shape, jnp.float32)


def _hook_loss(case):
    weights = _weights((DEPTH, B, PREFIX, WIDTH))

    def loss(variables, prefix):
        *_, layer_inputs = case["module"].apply(
            variables,
            [prefix, case["suffix"]],
            case["positions"],
            case["mask"],
            [None, case["cond"]],
            return_layer_inputs=True,
        )
        return jnp.sum(weights * layer_inputs)

    def reference(variables, prefix):
        inputs, _ = _layer_by_layer(
            variables, [prefix, case["suffix"]], case["positions"], case["mask"], [None, case["cond"]]
        )
        return jnp.sum(weights * inputs)

    return loss, reference


def test_remat_is_in_the_differentiated_program(case):
    loss, _ = _hook_loss(case)
    jaxpr = str(jax.make_jaxpr(jax.grad(loss, argnums=(0, 1)))(case["variables"], case["prefix"]))
    assert "checkpoint" in jaxpr or "remat" in jaxpr


@pytest.mark.parametrize("jit", [False, True])
def test_gradient_through_the_layer_inputs_is_finite_and_correct(case, jit):
    loss, reference = _hook_loss(case)
    grad = jax.grad(loss, argnums=(0, 1))
    ref_grad = jax.grad(reference, argnums=(0, 1))
    if jit:
        grad = jax.jit(grad)
    (g_vars, g_prefix), (r_vars, r_prefix) = (
        grad(case["variables"], case["prefix"]),
        ref_grad(case["variables"], case["prefix"]),
    )
    for leaf in jax.tree.leaves((g_vars, g_prefix)):
        assert bool(jnp.all(jnp.isfinite(leaf)))
    assert float(jnp.max(jnp.abs(g_prefix))) > 0
    _close(np.asarray(g_prefix), np.asarray(r_prefix))
    for got, want in zip(jax.tree.leaves(g_vars), jax.tree.leaves(r_vars), strict=True):
        _close(np.asarray(got), np.asarray(want))

    # Layer i's input depends on layers 0..i-1 only: the last layer and the suffix expert get exactly zero gradient,
    # every other first-expert layer a nonzero one.
    for path, leaf in jax.tree_util.tree_flatten_with_path(g_vars["params"]["layers"])[0]:
        if any(key.key.endswith("_1") for key in path):
            assert not bool(jnp.any(leaf)), path
        else:
            assert not bool(jnp.any(leaf[DEPTH - 1])), path
            assert bool(jnp.any(leaf[: DEPTH - 1])), path


# ---- static_argnums: flax counts `self` as index 0 ----


def test_static_argnums_name_exactly_the_python_flags_of_block_call():
    """The remat'd Block's static_argnums, read from the live Module.setup, are the indices (self = 0) of exactly the
    arguments that drive Python control flow, slicing or indexing: not whatever argument the number would point at in a
    self-less count. The probe goes through nn.remat itself, so it cannot disagree with what the scan uses."""
    names = list(inspect.signature(gemma.Block.__call__).parameters)
    assert names[0] == "self"
    static_names = {
        "deterministic",
        "collect_attention",
        "attention_key_count",
        "pool_expert",
        "return_layer_inputs",
        "layer_input_slice",
    }
    module = _gen.gemma_module(INPUTS)
    with mock.patch.object(gemma.nn, "remat", wraps=nn.remat) as spy:
        nn.Module.init(
            module,
            jax.random.key(0),
            [INPUTS["prefix"], INPUTS["suffix"]],
            INPUTS["positions"],
            INPUTS["mask"],
            [None, INPUTS["cond"]],
        )
    spy.assert_called()
    used = spy.call_args.kwargs["static_argnums"]
    assert sorted(names[i] for i in used) == sorted(static_names)


def test_static_flags_are_python_values_under_jit_and_dropout(case):
    """deterministic (index 6) selects Python control flow in nn.Dropout, collect_attention (7) and
    return_layer_inputs (13) pick the return structure, attention_key_count (9) and layer_input_slice (14) are used
    as slice bounds and pool_expert (12) indexes a list: every one of them has to arrive as a static Python value
    under the remat, so each is driven here through a jitted call (the arrays traced, the flags not)."""
    module = _gen.gemma_module(INPUTS, dropout=0.3)
    variables = _gen.gemma_variables(INPUTS, dropout=0.3)

    @jax.jit
    def run(variables, prefix, key):
        args = ([prefix, None], case["prefix_positions"], case["prefix_mask"])
        train = module.apply(variables, *args, deterministic=False, rngs={"dropout": key})[0][0]
        evaluated, _, attention, pooled, layer_inputs = module.apply(
            variables,
            *args,
            deterministic=True,
            collect_attention=True,
            attention_query_indices=case["query_indices"],
            attention_key_count=PREFIX,
            pool_mask=case["prefix_pool"],
            pool_expert=0,
            return_layer_inputs=True,
            layer_input_slice=(1, 4),
        )
        return train, evaluated[0], attention, pooled, layer_inputs

    train, evaluated, attention, pooled, layer_inputs = run(variables, case["prefix"], jax.random.key(1))
    assert float(jnp.max(jnp.abs(train - evaluated))) > 1e-3, "deterministic=False really dropped something"
    assert attention.shape == (DEPTH, B, 2, PREFIX)
    assert pooled.shape == (DEPTH, B, WIDTH)
    assert layer_inputs.shape == (DEPTH, B, 4, WIDTH)


_STATIC = {
    6: "deterministic",
    7: "collect_attention",
    9: "attention_key_count",
    12: "pool_expert",
    13: "return_layer_inputs",
    14: "layer_input_slice",
}


@pytest.mark.parametrize("dropped", sorted(_STATIC), ids=lambda i: _STATIC[i])
def test_every_static_index_is_needed(dropped):
    """Leave one index out of static_argnums and a call that uses every flag cannot be made: each of the six has to be
    static (measured: deterministic, collect_attention and return_layer_inputs fail with a traced bool, pool_expert
    with a traced int, attention_key_count and layer_input_slice with a traced slice bound). So the full tuple is not
    padded, and the passing runs above are not vacuous."""
    variables = _gen.gemma_variables(INPUTS, dropout=0.3)
    real_remat = nn.remat

    def remat_without(block, **kwargs):
        kwargs["static_argnums"] = tuple(i for i in _STATIC if i != dropped)
        return real_remat(block, **kwargs)

    def call():
        return _gen.gemma_module(INPUTS, dropout=0.3).apply(
            variables,
            [INPUTS["prefix"], None],
            INPUTS["prefix_positions"],
            INPUTS["prefix_mask"],
            deterministic=False,
            rngs={"dropout": jax.random.key(1)},
            collect_attention=True,
            attention_query_indices=INPUTS["query_indices"],
            attention_key_count=PREFIX,
            pool_mask=INPUTS["prefix_pool"],
            pool_expert=0,
            return_layer_inputs=True,
            layer_input_slice=(1, 3),
        )

    assert call()[-1].shape == (DEPTH, B, 3, WIDTH)
    with mock.patch.object(gemma.nn, "remat", remat_without), pytest.raises((jax.errors.JAXTypeError, IndexError)):
        call()


def test_dropout_needs_the_static_deterministic_flag():
    module = _gen.gemma_module(INPUTS, dropout=0.3)
    args = ([INPUTS["prefix"], INPUTS["suffix"]], INPUTS["positions"], INPUTS["mask"], [None, INPUTS["cond"]])
    variables = nn.Module.init(module, jax.random.key(0), *args)
    rngs = {"dropout": jax.random.key(1)}
    kept = module.apply(variables, *args, deterministic=True)[0][0]
    dropped = module.apply(variables, *args, deterministic=False, rngs=rngs)[0][0]
    assert float(jnp.max(jnp.abs(kept - dropped))) > 1e-3
    again = module.apply(variables, *args, deterministic=False, rngs=rngs)[0][0]
    np.testing.assert_array_equal(np.asarray(dropped), np.asarray(again))
    *_, layer_inputs = module.apply(variables, *args, deterministic=False, rngs=rngs, return_layer_inputs=True)
    assert layer_inputs.shape == (DEPTH, B, PREFIX, WIDTH)


# ---- a cache that already holds extra (latent) K/V columns ----


def _rope(x, positions):
    half = x.shape[-1] // 2
    timescale = 10_000.0 ** ((2.0 / x.shape[-1]) * np.arange(half))
    radians = (positions[..., None] / timescale)[:, :, None, :]
    sin, cos = np.sin(radians), np.cos(radians)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def _numpy_attention(params, xs, positions, mask, cache):
    """Float64 numpy attention of both experts over [cache; new keys]: independent of gemma.Attention."""

    def f64(a):
        return np.asarray(a, np.float64)

    qs, ks, vs = [], [], []
    for i, x in enumerate(xs):
        suffix = "" if i == 0 else f"_{i}"
        qs.append(np.einsum("btd,ndh->btnh", f64(x), f64(params[f"q_einsum{suffix}"]["w"])))
        k, v = np.einsum("bsd,ckdh->cbskh", f64(x), f64(params[f"kv_einsum{suffix}"]["w"]))
        ks.append(k)
        vs.append(v)
    q, k, v = (np.concatenate(a, axis=1) for a in (qs, ks, vs))
    pos = f64(positions)
    q = _rope(q, pos) * HEAD_DIM**-0.5
    k = _rope(k, pos)
    if cache is not None:
        k, v = np.concatenate([f64(cache[0]), k], axis=1), np.concatenate([f64(cache[1]), v], axis=1)
    logits = np.einsum("btnh,bskh->bnts", q, k)  # one kv head: the query heads all share it
    logits = np.where(np.asarray(mask)[:, None], logits, -np.inf)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    encoded = np.einsum("bnts,bskh->btnh", probs, v)
    outs, start = [], 0
    for i, x in enumerate(xs):
        suffix = "" if i == 0 else f"_{i}"
        outs.append(
            np.einsum(
                "btnh,nhd->btd", encoded[:, start : start + x.shape[1]], f64(params[f"attn_vec_einsum{suffix}"]["w"])
            )
        )
        start += x.shape[1]
    return outs, (k, v)


def _attention_case(extra):
    rng = np.random.default_rng(1)

    def normal(*shape):
        return jnp.asarray(rng.normal(size=shape), jnp.float32)

    cached = 4
    xs = [normal(B, 2, WIDTH), normal(B, SUFFIX, WIDTH)]
    new = 2 + SUFFIX
    positions = jnp.broadcast_to(cached + jnp.arange(new), (B, new))
    base = (normal(B, cached, 1, HEAD_DIM), normal(B, cached, 1, HEAD_DIM))
    latents = (normal(B, extra, 1, HEAD_DIM), normal(B, extra, 1, HEAD_DIM))
    extended = tuple(jnp.concatenate([b, latent], axis=1) for b, latent in zip(base, latents, strict=True))
    attn = gemma.Attention(configs=CONFIGS)
    params = nn.Module.init(attn, jax.random.key(0), xs, positions, jnp.ones((B, 1, new, new), bool), None)["params"]
    return attn, params, xs, positions, base, extended, new


def test_attention_over_a_cache_with_extra_columns_equals_attention_over_the_concatenation():
    attn, params, xs, positions, _, extended, new = _attention_case(LATENTS)
    columns = extended[0].shape[1] + new
    rng = np.random.default_rng(2)
    mask = rng.random((B, new, columns)) < 0.7
    mask[:, :, 0] = True  # every query sees something
    (out0, out1), (k, v) = attn.apply({"params": params}, xs, positions, jnp.asarray(mask)[:, None], extended)
    expected, (expected_k, expected_v) = _numpy_attention(params, xs, positions, mask, extended)
    _close(np.concatenate([out0, out1], axis=1), np.concatenate(expected, axis=1))
    # the returned cache is the whole key axis: the caller's columns first, then this call's keys
    assert k.shape == v.shape == (B, columns, 1, HEAD_DIM)
    np.testing.assert_array_equal(np.asarray(k[:, : extended[0].shape[1]]), np.asarray(extended[0]))
    np.testing.assert_array_equal(np.asarray(v[:, : extended[1].shape[1]]), np.asarray(extended[1]))
    _close(np.asarray(k), expected_k)
    _close(np.asarray(v), expected_v)


def test_extra_columns_that_are_masked_out_change_nothing():
    attn, params, xs, positions, base, extended, new = _attention_case(LATENTS)
    rng = np.random.default_rng(3)
    plain_mask = rng.random((B, new, base[0].shape[1] + new)) < 0.7
    plain_mask[:, :, 0] = True
    cached = base[0].shape[1]
    wide_mask = np.concatenate(
        [plain_mask[:, :, :cached], np.zeros((B, new, LATENTS), bool), plain_mask[:, :, cached:]], axis=-1
    )
    plain, _ = attn.apply({"params": params}, xs, positions, jnp.asarray(plain_mask)[:, None], base)
    wide, _ = attn.apply({"params": params}, xs, positions, jnp.asarray(wide_mask)[:, None], extended)
    for a, b in zip(plain, wide, strict=True):
        _close(np.asarray(a), np.asarray(b))


def test_the_mask_must_cover_the_extra_columns():
    attn, params, xs, positions, base, extended, new = _attention_case(LATENTS)
    too_narrow = jnp.ones((B, 1, new, base[0].shape[1] + new), bool)
    with pytest.raises(ValueError, match="Attention mask"):
        attn.apply({"params": params}, xs, positions, too_narrow, extended)


def test_stacked_cache_with_latent_columns_through_the_scan(case):
    """The shape LIT uses: the prefix pass's stacked (layers, b, s, kv, h) cache, per-layer latent K/V concatenated on
    the key axis, a (b, t, s + latents + t) mask, and the suffix expert alone running over it."""
    _, prefix_cache = _prefix_only(case)
    rng = np.random.default_rng(4)
    latents = tuple(jnp.asarray(rng.normal(size=(DEPTH, B, LATENTS, 1, HEAD_DIM)), jnp.float32) for _ in range(2))
    extended = tuple(jnp.concatenate([c, latent], axis=2) for c, latent in zip(prefix_cache, latents, strict=True))
    mask = np.ones((B, SUFFIX, PREFIX + LATENTS + SUFFIX), bool)
    mask[1, :, PREFIX - 2 : PREFIX] = False
    mask[0, 1, PREFIX + 1] = False  # one latent column hidden from one row
    mask = jnp.asarray(mask)
    args = ([None, case["suffix"]], case["suffix_positions"], mask, [None, case["cond"]])

    (out, _) = case["module"].apply(case["variables"], *args, kv_cache=extended)

    block = gemma.Block(configs=tuple(CONFIGS))
    xs = [None, case["suffix"]]
    for i in range(DEPTH):
        cache_i = jax.tree.map(lambda a, i=i: a[i], extended)
        xs, _ = block.apply(
            {"params": _layer_params(case["variables"], i)},
            xs,
            cache_i,
            case["suffix_positions"],
            mask[:, None],
            [None, case["cond"]],
            deterministic=True,
        )
    expected = gemma.RMSNorm().apply({"params": case["variables"]["params"]["final_norm_1"]}, xs[1], case["cond"])[0]
    _close(np.asarray(out[1]), np.asarray(expected))

    # and the latent columns matter: hiding all of them gives a different answer
    hidden = mask.at[:, :, PREFIX : PREFIX + LATENTS].set(False)
    (out_hidden, _) = case["module"].apply(
        case["variables"],
        [None, case["suffix"]],
        case["suffix_positions"],
        hidden,
        [None, case["cond"]],
        kv_cache=extended,
    )
    assert float(jnp.max(jnp.abs(out[1] - out_hidden[1]))) > 1e-4
