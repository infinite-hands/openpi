"""Tests for the LIT interface modules (lit.py).

Golden parity: lit_golden/torch_interface.npz holds the weights, inputs and outputs of the Apache-2.0 MolmoAct2 LIT fork's
torch classes at tiny dimensions (lit_golden/gen_torch_fixture.py, NOTICE). The oracle proves equivalence to that module,
not to the unlicensed pi0.5 plugin; the real-dimension comparison against the released pi0.5 header is informational.
"""

import pathlib

import flax.nnx as nnx
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import lit

GOLDEN = pathlib.Path(__file__).parent / "lit_golden"

# Measured with measure_parity() on CPU. float32: the largest JAX-vs-torch error is 1.0e-5 (outputs reach 16), the same as the
# torch float32-vs-float64 rounding floor (9.5e-6), so the two implementations are indistinguishable from rounding; the
# tolerance is 1e-4, about 10x that. It still rejects real semantic differences, measured by mutating lit.py: tanh GELU
# 7.2e-3, LayerNorm eps 1e-6 5.4e-4, swapped semantic/image masks 19, interleaved head split 11. float64: the largest error is
# 2.2e-14, so 1e-12 leaves ~45x.
ATOL_F32 = 1e-4
ATOL_F64 = 1e-12


@pytest.fixture(scope="module")
def fx():
    with np.load(GOLDEN / "torch_interface.npz", allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def _dims(fx):
    return {k[len("dims/") :]: fx[k].item() for k in fx if k.startswith("dims/")}


def _linear(w):
    return w.T


def _split3(w, d):
    return w[:d].T, w[d : 2 * d].T, w[2 * d :].T


def _dense(w, b):
    return {"kernel": _linear(w), "bias": b}


def _norm(sd, name):
    return {"scale": sd[f"{name}.weight"], "bias": sd[f"{name}.bias"]}


def _ffn(sd, name):
    return {
        "up": _dense(sd[f"{name}.0.weight"], sd[f"{name}.0.bias"]),
        "down": _dense(sd[f"{name}.3.weight"], sd[f"{name}.3.bias"]),
    }


def _attn_self(sd, name, d):
    wq, wk, wv = _split3(sd[f"{name}.in_proj_weight"], d)
    bq, bk, bv = np.split(sd[f"{name}.in_proj_bias"], 3)
    return {
        "q": {"kernel": wq, "bias": bq},
        "k": {"kernel": wk, "bias": bk},
        "v": {"kernel": wv, "bias": bv},
        "o": _dense(sd[f"{name}.out_proj.weight"], sd[f"{name}.out_proj.bias"]),
    }


def _attn_cross(sd, name):
    bq, bk, bv = np.split(sd[f"{name}.in_proj_bias"], 3)
    return {
        "q": {"kernel": _linear(sd[f"{name}.q_proj_weight"]), "bias": bq},
        "k": {"kernel": _linear(sd[f"{name}.k_proj_weight"]), "bias": bk},
        "v": {"kernel": _linear(sd[f"{name}.v_proj_weight"]), "bias": bv},
        "o": _dense(sd[f"{name}.out_proj.weight"], sd[f"{name}.out_proj.bias"]),
    }


def _torch_group_to_jax(sd, d):
    """One torch group (keys relative to the group, e.g. 'self_block.attn_norm.weight') -> our per-group param tree."""

    def cross(name):
        return {
            "query_norm": _norm(sd, f"{name}.query_norm"),
            "context_norm": _norm(sd, f"{name}.context_norm"),
            "attn": _attn_cross(sd, f"{name}.cross_attn"),
            "ffn_norm": _norm(sd, f"{name}.ffn_norm"),
            "ffn": _ffn(sd, f"{name}.ffn"),
        }

    return {
        "self_block": {
            "norm": _norm(sd, "self_block.attn_norm"),
            "attn": _attn_self(sd, "self_block.self_attn", d),
            "ffn_norm": _norm(sd, "self_block.ffn_norm"),
            "ffn": _ffn(sd, "self_block.ffn"),
        },
        "semantic_block": cross("semantic_block"),
        "visual_block": cross("visual_block"),
        "to_key": {"kernel": _linear(sd["to_key.weight"])},
        "to_value": {"kernel": _linear(sd["to_value.weight"])},
    }


def torch_aggregator_to_jax(w, d):
    """The torch aggregator state dict (keys 'w/agg/...' in the fixture) -> our LitAggregator param tree.

    Torch holds group 0 at the top level and groups 1.. under additional_groups.{g-1}; we stack all groups on axis 0.
    """
    sd = {k[len("w/agg/") :]: v for k, v in w.items() if k.startswith("w/agg/")}
    groups = []
    for g in range(d["num_groups"]):
        prefix = "" if g == 0 else f"additional_groups.{g - 1}."
        group_sd = {
            k[len(prefix) :]: v for k, v in sd.items() if (k.startswith(prefix) if g else "additional_groups" not in k)
        }
        groups.append(_torch_group_to_jax(group_sd, d["dim"]))
    return {"queries": sd["queries"], "groups": jax.tree.map(lambda *xs: np.stack(xs), *groups)}


def torch_pose_to_jax(w):
    return {
        "norm": {"scale": w["w/pose_norm/weight"], "bias": w["w/pose_norm/bias"]},
        **{
            f"fc{i + 1}": _dense(w[f"w/pose_decoder/net.{2 * i}.weight"], w[f"w/pose_decoder/net.{2 * i}.bias"])
            for i in range(3)
        },
    }


def torch_goal_to_jax(w):
    return {
        **{
            f"fc{i + 1}": _dense(w[f"w/goal_encoder/net.{2 * i}.weight"], w[f"w/goal_encoder/net.{2 * i}.bias"])
            for i in range(3)
        },
        "to_key": {"kernel": _linear(w["w/goal_to_key/weight"])},
        "to_value": {"kernel": _linear(w["w/goal_to_value/weight"])},
    }


def load_params(module, tree, dtype):
    """Overwrite every Param of `module` from the nested dict `tree`; the key sets and shapes must match exactly."""
    flat = nnx.state(module, nnx.Param).flat_state()
    paths = {tuple(p): v for p, v in flat.items()}
    given = {}

    def walk(prefix, node):
        if isinstance(node, dict):
            for k, v in node.items():
                walk((*prefix, k), v)
        else:
            given[prefix] = node

    walk((), tree)
    assert set(paths) == set(given), (sorted(set(paths) - set(given)), sorted(set(given) - set(paths)))
    for path, arr in given.items():
        assert paths[path].value.shape == arr.shape, (path, paths[path].value.shape, arr.shape)
        obj = module
        for key in path[:-1]:
            obj = getattr(obj, key)
        getattr(obj, path[-1]).value = jnp.asarray(arr, dtype)


def build_from_fixture(fx, dtype=jnp.float32):
    d = _dims(fx)
    rngs = nnx.Rngs(0)
    agg = lit.LitAggregator(
        num_latents=d["num_latents"],
        dim=d["dim"],
        context_dim=d["context_dim"],
        kv_dim=d["kv_dim"],
        num_heads=d["num_heads"],
        ffn_ratio=d["ffn_ratio"],
        num_groups=d["num_groups"],
        rngs=rngs,
        dtype=dtype,
        param_dtype=dtype,
    )
    pose = lit.LitPoseDecoder(
        pose_dim=d["pose_dim"],
        num_tokens=d["pose_tokens"],
        dim=d["dim"],
        inner_dim=d["inner_dim"],
        rngs=rngs,
        param_dtype=dtype,
    )
    goal = lit.LitGoalEncoder(
        pose_dim=d["pose_dim"],
        num_tokens=d["goal_tokens"],
        dim=d["dim"],
        kv_dim=d["kv_dim"],
        inner_dim=d["inner_dim"],
        rngs=rngs,
        param_dtype=dtype,
    )
    load_params(agg, torch_aggregator_to_jax(fx, d), dtype)
    load_params(pose, torch_pose_to_jax(fx), dtype)
    load_params(goal, torch_goal_to_jax(fx), dtype)
    return agg, pose, goal


def run_jax(fx, dtype=jnp.float32):
    """Forward the fixture inputs; returns the same named outputs as the torch fixture (both call styles of the aggregator)."""
    agg, pose, goal = build_from_fixture(fx, dtype)
    hid = jnp.asarray(fx["in/layer_hiddens"], dtype)
    sem, img = jnp.asarray(fx["in/semantic_mask"]), jnp.asarray(fx["in/image_mask"])
    num_layers = hid.shape[0]

    final, keys, values = agg(hid, sem, img)
    lat = agg.initial_latents(hid.shape[1])
    per_layer, step_k, step_v = [], [], []
    for layer in range(num_layers):
        g = agg.group_index(layer, num_layers)
        lat = agg.step(lat, hid[layer], sem, img, g)
        k, v = agg.project_kv(lat, g)
        per_layer.append(lat), step_k.append(k), step_v.append(v)
    goal_tokens = goal(jnp.asarray(fx["in/goal"], dtype))
    goal_k, goal_v = goal.project_kv(goal_tokens)
    return {
        "latents_per_layer": jnp.stack(per_layer),
        "keys": keys,
        "values": values,
        "final_latents": final,
        "pose_pred": pose(final),
        "goal_tokens": goal_tokens,
        "goal_key": goal_k,
        "goal_value": goal_v,
        "step_final": lat,
        "step_keys": jnp.stack(step_k),
        "step_values": jnp.stack(step_v),
    }


COMPARED = (
    "latents_per_layer",
    "keys",
    "values",
    "final_latents",
    "pose_pred",
    "goal_tokens",
    "goal_key",
    "goal_value",
)


def measure_parity(fx):
    """max |error| per output: JAX f32 vs torch f32, JAX f32 vs torch f64, torch f32 vs torch f64 (the rounding floor)."""
    out = run_jax(fx)
    rows = {}
    for name in COMPARED:
        j, t32, t64 = np.asarray(out[name], np.float64), fx[f"out32/{name}"].astype(np.float64), fx[f"out64/{name}"]
        rows[name] = (np.abs(j - t32).max(), np.abs(j - t64).max(), np.abs(t32 - t64).max(), np.abs(t64).max())
    return rows


def test_torch_parity_float32(fx):
    out = run_jax(fx)
    for name in COMPARED:
        np.testing.assert_allclose(np.asarray(out[name]), fx[f"out32/{name}"], rtol=0, atol=ATOL_F32, err_msg=name)
    # The per-layer step chain and the scan over all layers are the same computation.
    np.testing.assert_allclose(out["step_final"], out["final_latents"], rtol=0, atol=ATOL_F32)
    np.testing.assert_allclose(out["step_keys"], out["keys"], rtol=0, atol=ATOL_F32)
    np.testing.assert_allclose(out["step_values"], out["values"], rtol=0, atol=ATOL_F32)


def test_torch_parity_float64(fx):
    with jax.experimental.enable_x64():
        out = run_jax(fx, jnp.float64)
        assert out["keys"].dtype == jnp.float64
        for name in COMPARED:
            np.testing.assert_allclose(np.asarray(out[name]), fx[f"out64/{name}"], rtol=0, atol=ATOL_F64, err_msg=name)


def test_fixture_outputs_are_not_degenerate(fx):
    # Guards the oracle itself: outputs are O(1) (not all-zero), latents change from layer to layer, and the groups differ.
    for name in COMPARED:
        assert np.abs(fx[f"out64/{name}"]).max() > 0.1, name
    lpl = fx["out64/latents_per_layer"]
    assert all(np.abs(lpl[i + 1] - lpl[i]).max() > 0.1 for i in range(lpl.shape[0] - 1))
    sem, img = fx["in/semantic_mask"], fx["in/image_mask"]
    assert sem.any(1).all()
    assert img.any(1).all()
    assert (~(sem | img)).any()


def test_layer_group_map(fx):
    assert [lit.layer_group_index(i, 18, 6) for i in range(18)] == [i // 3 for i in range(18)]
    d = _dims(fx)
    torch_map = [lit.layer_group_index(i, d["num_layers"], d["num_groups"]) for i in range(d["num_layers"])]
    assert torch_map == fx["out32/group_index"].tolist()
    for bad_layers, bad_groups in ((18, 5), (17, 6), (0, 1), (-6, 6), (6, 0)):
        with pytest.raises(ValueError, match="num_"):
            lit.layer_group_index(0, bad_layers, bad_groups)
    for bad_idx in (-1, 18):
        with pytest.raises(IndexError, match="layer_idx"):
            lit.layer_group_index(bad_idx, 18, 6)
    agg, _, _ = build_from_fixture(fx)
    with pytest.raises(ValueError, match="divisible"):
        agg(jnp.zeros((4, 2, 5, 48)), jnp.ones((2, 5), bool), jnp.ones((2, 5), bool))  # 4 layers, 3 groups


def test_role_masks():
    roles = jnp.array([[lit.ROLE_IMAGE, lit.ROLE_SEMANTIC, lit.ROLE_PAD]])
    sem, img = lit.role_masks(roles)
    assert sem.tolist() == [[False, True, False]]
    assert img.tolist() == [[True, False, False]]


def test_masked_context_columns_change_nothing(fx):
    agg, _, _ = build_from_fixture(fx)
    hid = jnp.asarray(fx["in/layer_hiddens"])
    sem, img = jnp.asarray(fx["in/semantic_mask"]), jnp.asarray(fx["in/image_mask"])
    pad = ~(sem | img)
    assert bool(pad.any())
    base = agg(hid, sem, img)
    rng = np.random.RandomState(0)
    junk = jnp.asarray(rng.randn(*hid.shape).astype(np.float32) * 1e3)
    for poison in (junk, jnp.full_like(hid, jnp.nan), jnp.full_like(hid, jnp.inf)):
        out = agg(jnp.where(pad[None, :, :, None], poison, hid), sem, img)
        for a, b in zip(base, out, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # Unmasked columns are live: changing an image column changes the latents.
    live = jnp.where(img[None, :, :, None], hid + junk[..., : hid.shape[-1]] * 1e-3, hid)
    assert float(jnp.abs(agg(live, sem, img)[0] - base[0]).max()) > 1e-3


def test_each_stream_reads_only_its_own_columns(fx):
    agg, _, _ = build_from_fixture(fx)
    hid = jnp.asarray(fx["in/layer_hiddens"][0])
    sem, img = jnp.asarray(fx["in/semantic_mask"]), jnp.asarray(fx["in/image_mask"])
    p = jax.tree.map(lambda a: a[0], agg.group_params())
    x = agg.initial_latents(hid.shape[0])
    noise = jnp.asarray(np.random.RandomState(4).randn(*hid.shape).astype(np.float32))
    for block, own in (("semantic_block", sem), ("visual_block", img)):
        base = lit._cross_block(p[block], x, hid, own, agg.num_heads)  # noqa: SLF001
        moved = lit._cross_block(p[block], x, jnp.where(own[..., None], hid, hid + noise), own, agg.num_heads)  # noqa: SLF001
        np.testing.assert_array_equal(np.asarray(base), np.asarray(moved))
        moved = lit._cross_block(p[block], x, jnp.where(own[..., None], hid + noise, hid), own, agg.num_heads)  # noqa: SLF001
        assert float(jnp.abs(moved - base).max()) > 1e-3


def test_all_masked_context_is_finite_and_independent_of_the_context(fx):
    agg, pose, _ = build_from_fixture(fx)
    hid = jnp.asarray(fx["in/layer_hiddens"])
    sem, img = jnp.asarray(fx["in/semantic_mask"]), jnp.asarray(fx["in/image_mask"])
    none = jnp.zeros_like(sem)
    for s, i in ((none, none), (none, img), (sem, none)):
        out = agg(hid, s, i)
        assert all(bool(jnp.isfinite(x).all()) for x in out)
        assert bool(jnp.isfinite(pose(out[0])).all())
    # A row with no context ignores it entirely, and its neighbours in the batch are not affected by it.
    s, i = sem.at[0].set(False), img.at[0].set(False)
    a = agg(hid, s, i)
    b = agg(hid.at[:, 0].set(hid[:, 0] * -7.0 + 3.0), s, i)
    np.testing.assert_array_equal(np.asarray(a[0][0]), np.asarray(b[0][0]))
    np.testing.assert_array_equal(np.asarray(agg(hid, sem, img)[0][1:]), np.asarray(a[0][1:]))


def test_empty_row_attention_output_is_the_output_bias(fx):
    agg, _, _ = build_from_fixture(fx)
    p = jax.tree.map(lambda a: a[0], agg.group_params())["semantic_block"]
    x = jnp.asarray(np.random.RandomState(1).randn(2, 4, 32).astype(np.float32))
    ctx = jnp.asarray(np.random.RandomState(2).randn(2, 5, 48).astype(np.float32))
    out = lit._attend(  # noqa: SLF001
        p["attn"], x, ctx, jnp.zeros((2, 5), bool), 4
    )
    np.testing.assert_allclose(out, jnp.broadcast_to(p["attn"]["o"]["bias"], out.shape), atol=1e-6)


def _loss(agg, pose, goal, hid, sem, img, goal_in, w):
    final, keys, values = agg(hid, sem, img)
    gk, gv = goal.project_kv(goal(goal_in))
    return (
        (keys * w["k"]).sum()
        + (values * w["v"]).sum()
        + (pose(final) * w["pose"]).sum()
        + (gk * w["gk"]).sum()
        + (gv * w["gv"]).sum()
    )


def _tiny_models(fx, *, remat=False):
    d = _dims(fx)
    rngs = nnx.Rngs(3)
    agg = lit.LitAggregator(
        num_latents=d["num_latents"],
        dim=d["dim"],
        context_dim=d["context_dim"],
        kv_dim=d["kv_dim"],
        num_heads=d["num_heads"],
        ffn_ratio=d["ffn_ratio"],
        num_groups=d["num_groups"],
        rngs=rngs,
        remat=remat,
    )
    pose = lit.LitPoseDecoder(
        pose_dim=d["pose_dim"], num_tokens=d["pose_tokens"], dim=d["dim"], inner_dim=d["inner_dim"], rngs=rngs
    )
    goal = lit.LitGoalEncoder(
        pose_dim=d["pose_dim"],
        num_tokens=d["goal_tokens"],
        dim=d["dim"],
        kv_dim=d["kv_dim"],
        inner_dim=d["inner_dim"],
        rngs=rngs,
    )
    return agg, pose, goal


def _grad_inputs(fx):
    rng = np.random.RandomState(5)
    d = _dims(fx)
    shapes = {
        "k": (d["num_layers"], d["batch"], d["num_latents"], d["kv_dim"]),
        "v": (d["num_layers"], d["batch"], d["num_latents"], d["kv_dim"]),
        "pose": (d["batch"], d["pose_dim"]),
        "gk": (d["batch"], d["goal_tokens"], d["kv_dim"]),
        "gv": (d["batch"], d["goal_tokens"], d["kv_dim"]),
    }
    w = {k: jnp.asarray(rng.randn(*s).astype(np.float32)) for k, s in shapes.items()}
    return (
        jnp.asarray(fx["in/layer_hiddens"]),
        jnp.asarray(fx["in/semantic_mask"]),
        jnp.asarray(fx["in/image_mask"]),
        jnp.asarray(fx["in/goal"]),
        w,
    )


def test_gradients_are_finite_and_nonzero_for_every_group(fx):
    models = _tiny_models(fx)
    hid, sem, img, goal_in, w = _grad_inputs(fx)
    grads = nnx.grad(lambda a, p, g: _loss(a, p, g, hid, sem, img, goal_in, w), argnums=(0, 1, 2))(*models)
    scale = max(float(jnp.abs(leaf).max()) for leaf in jax.tree.leaves(grads))
    assert scale > 0
    zero_by_construction = 0
    for name, g_tree in zip(("aggregator", "pose_decoder", "goal_encoder"), grads, strict=True):
        for path, var in g_tree.flat_state().items():
            g = np.asarray(var.value)
            key = "/".join(map(str, path))
            assert np.isfinite(g).all(), (name, key)
            slices = g if name == "aggregator" and path[0] == "groups" else g[None]
            if path[-2:] == ("k", "bias"):
                # softmax is invariant to a constant shift of every key logit, so the key bias has exactly zero gradient.
                assert np.abs(g).max() < 1e-6 * scale, (name, key, np.abs(g).max())
                zero_by_construction += 1
                continue
            for i, s in enumerate(slices):
                assert np.abs(s).max() > 0, (name, key, i)
    assert zero_by_construction == 3  # the key bias of the self, semantic and visual attention


def test_gradients_finite_with_empty_context_rows(fx):
    models = _tiny_models(fx)
    hid, sem, img, goal_in, w = _grad_inputs(fx)
    sem, img = sem.at[0].set(False), img.at[1].set(False)
    grads = nnx.grad(lambda a, p, g: _loss(a, p, g, hid, sem, img, goal_in, w), argnums=(0, 1, 2))(*models)
    assert all(bool(jnp.isfinite(leaf).all()) for leaf in jax.tree.leaves(grads))


def test_pose_loss_reaches_aggregator_and_decoder_only(fx):
    agg, pose, goal = _tiny_models(fx)
    hid, sem, img, _, _ = _grad_inputs(fx)
    target = jnp.ones((hid.shape[1], _dims(fx)["pose_dim"]))
    grads = nnx.grad(lambda a, p: jnp.mean((p(a(hid, sem, img)[0]) - target) ** 2), argnums=(0, 1))(agg, pose)
    for name, tree in zip(("aggregator", "pose_decoder"), grads, strict=True):
        for path, var in tree.flat_state().items():
            g = np.asarray(var.value)
            if path[-2:] == ("k", "bias"):
                continue
            if path[:2] in (("groups", "to_key"), ("groups", "to_value")):
                # They only feed the K/V, never the latents the pose decoder reads: no pose gradient, as designed.
                assert np.abs(g).max() == 0.0, path
                continue
            slices = g if path[0] == "groups" else g[None]
            for i, s in enumerate(slices):
                assert np.abs(s).max() > 0, (name, path, i)


def test_remat_matches(fx):
    plain, pose, _ = _tiny_models(fx)
    remat, _, _ = _tiny_models(fx, remat=True)
    hid, sem, img, _, w = _grad_inputs(fx)

    def loss(a):
        final, k, v = a(hid, sem, img)
        return (k * w["k"]).sum() + (v * w["v"]).sum() + final.sum()

    (v1, g1), (v2, g2) = nnx.value_and_grad(loss)(plain), nnx.value_and_grad(loss)(remat)
    np.testing.assert_allclose(v1, v2, rtol=1e-5)
    scale = max(float(jnp.abs(leaf).max()) for leaf in jax.tree.leaves(g1))
    for a, b in zip(jax.tree.leaves(g1), jax.tree.leaves(g2), strict=True):
        # Rounding differs, so compare against the largest gradient (the key-bias gradients are pure rounding noise).
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-4 * scale)


def test_bfloat16_forward_is_finite(fx):
    d = _dims(fx)
    rngs = nnx.Rngs(0)
    agg = lit.LitAggregator(
        num_latents=d["num_latents"],
        dim=d["dim"],
        context_dim=d["context_dim"],
        kv_dim=d["kv_dim"],
        num_heads=d["num_heads"],
        ffn_ratio=d["ffn_ratio"],
        num_groups=d["num_groups"],
        rngs=rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
    )
    pose = lit.LitPoseDecoder(
        pose_dim=d["pose_dim"],
        num_tokens=d["pose_tokens"],
        dim=d["dim"],
        inner_dim=d["inner_dim"],
        rngs=rngs,
        param_dtype=jnp.bfloat16,
    )
    goal = lit.LitGoalEncoder(
        pose_dim=d["pose_dim"],
        num_tokens=d["goal_tokens"],
        dim=d["dim"],
        kv_dim=d["kv_dim"],
        inner_dim=d["inner_dim"],
        rngs=rngs,
        param_dtype=jnp.bfloat16,
    )
    hid = jnp.asarray(fx["in/layer_hiddens"], jnp.bfloat16)
    sem, img = jnp.asarray(fx["in/semantic_mask"]), jnp.asarray(fx["in/image_mask"])
    final, keys, values = agg(hid, sem, img)
    assert final.dtype == keys.dtype == values.dtype == jnp.bfloat16
    pred = pose(final)
    tokens = goal(jnp.asarray(fx["in/goal"]))
    gk, gv = goal.project_kv(tokens.astype(jnp.bfloat16))
    for x in (final, keys, values, pred, tokens, gk, gv):
        assert bool(jnp.isfinite(x.astype(jnp.float32)).all())
    # float32 inputs into a bfloat16 model are cast, not rejected; empty rows stay finite in bfloat16 too.
    assert bool(jnp.isfinite(agg(hid.astype(jnp.float32), jnp.zeros_like(sem), img)[0].astype(jnp.float32)).all())


# ---- real dimensions vs the released pi0.5 LIT headers (informational) -----------------------------------------

_HEADER_POSE_DIM = 8  # the released LIBERO checkpoints' state dimension; ours is len(lit_goal_dims)


def _jax_shapes_real_dims():
    def make():
        rngs = nnx.Rngs(0)
        return {
            "agg": lit.LitAggregator(rngs=rngs),
            "pose": lit.LitPoseDecoder(pose_dim=_HEADER_POSE_DIM, rngs=rngs),
            "goal": lit.LitGoalEncoder(pose_dim=_HEADER_POSE_DIM, rngs=rngs),
        }

    abstract = nnx.eval_shape(make)
    shapes = {}
    for name, module in abstract.items():
        for path, var in nnx.state(module, nnx.Param).flat_state().items():
            shapes[(name, *path)] = tuple(var.value.shape)
    return shapes


def _expected_header_shapes(shapes, num_groups):
    """The tensors the released header would hold for our parameters (torch layout), keyed by header name.

    `shapes` are per-group shapes (the stacked leaves without their leading group axis).
    """
    out = {}
    out["model.aggregator.queries"] = shapes[("agg", "queries")]

    def t(shape):
        return (shape[1], shape[0])

    def norm(ours, header):
        out[f"{header}.weight"] = shapes[(*ours, "scale")][-1:]
        out[f"{header}.bias"] = shapes[(*ours, "bias")][-1:]

    def dense(ours, header, *, bias=True):
        out[f"{header}.weight"] = t(shapes[(*ours, "kernel")][-2:])
        if bias:
            out[f"{header}.bias"] = shapes[(*ours, "bias")][-1:]

    def ffn(ours, header):
        dense((*ours, "up"), f"{header}.0")
        dense((*ours, "down"), f"{header}.3")

    for g in range(num_groups):
        base = f"model.aggregator.groups.{g}"
        sb = ("agg", "groups", "self_block")
        norm((*sb, "norm"), f"{base}.self_block.norm")
        norm((*sb, "ffn_norm"), f"{base}.self_block.ffn_norm")
        ffn((*sb, "ffn"), f"{base}.self_block.ffn")
        (d,) = shapes[(*sb, "attn", "q", "bias")]
        out[f"{base}.self_block.attn.in_proj_weight"] = (3 * d, shapes[(*sb, "attn", "q", "kernel")][-2])
        out[f"{base}.self_block.attn.in_proj_bias"] = (3 * d,)
        dense((*sb, "attn", "o"), f"{base}.self_block.attn.out_proj")
        for block in ("semantic_block", "visual_block"):
            cb = ("agg", "groups", block)
            name = f"{base}.{block}"
            for n in ("query_norm", "context_norm", "ffn_norm"):
                norm((*cb, n), f"{name}.{n}")
            ffn((*cb, "ffn"), f"{name}.ffn")
            for p in "qkv":
                out[f"{name}.cross_attn.{p}_proj_weight"] = t(shapes[(*cb, "attn", p, "kernel")][-2:])
            out[f"{name}.cross_attn.in_proj_bias"] = (3 * d,)
            dense((*cb, "attn", "o"), f"{name}.cross_attn.out_proj")
        dense(("agg", "groups", "to_key"), f"{base}.to_key", bias=False)
        dense(("agg", "groups", "to_value"), f"{base}.to_value", bias=False)
    norm(("pose", "norm"), "model.pose_norm")
    for i in range(3):
        dense(("pose", f"fc{i + 1}"), f"model.pose_decoder.net.{2 * i}")
        dense(("goal", f"fc{i + 1}"), f"model.se3_encoder.net.{2 * i}")
    dense(("goal", "to_key"), "model.goal_to_key", bias=False)
    dense(("goal", "to_value"), "model.goal_to_value", bias=False)
    return out


def header_diff():
    """(header-only, ours-only, shape mismatches, total params ours, total params header) vs the released pi0.5 headers."""
    shapes = _jax_shapes_real_dims()
    # Stacked leaves carry a leading group axis; the per-group torch tensors drop it.
    per_group = {k: (v[1:] if k[:2] == ("agg", "groups") else v) for k, v in shapes.items()}
    ours = _expected_header_shapes(per_group, shapes[("agg", "groups", "to_key", "kernel")][0])
    with np.load(GOLDEN / "torch_pi05_header_shapes.npz", allow_pickle=False) as f:
        header = {str(n): tuple(int(x) for x in s if x >= 0) for n, s in zip(f["names"], f["shapes"], strict=True)}
    mismatched = {n: (ours[n], header[n]) for n in ours.keys() & header.keys() if ours[n] != header[n]}
    total_ours = sum(int(np.prod(s)) for s in shapes.values())
    total_header = sum(int(np.prod(s)) for s in header.values())
    return (
        sorted(header.keys() - ours.keys()),
        sorted(ours.keys() - header.keys()),
        mismatched,
        total_ours,
        total_header,
    )


def test_real_dimensions_against_the_released_pi05_header():
    header_only, ours_only, mismatched, total_ours, total_header = header_diff()
    print("\nheader-only tensors:", header_only)
    print("ours-only tensors:", ours_only)
    print("shape mismatches:", mismatched)
    print("params ours / header:", total_ours, total_header)
    assert not mismatched
    assert not ours_only
    # The only released tensors we have no parameter for are the Syn-gate (gate_init buffer, gate_delta), which the released
    # Stage-2 config switches off (use_syn_gate=false); 12 floats.
    assert header_only == ["model.aggregator.gate_delta", "model.aggregator.gate_init"]
    assert total_header - total_ours == 12


def test_real_dimension_forward_shapes():
    # eval_shape only: no 3B-parameter instantiation. 18 layers x (batch 2, one 3-camera block of 776 tokens) x width 2048.
    def forward():
        rngs = nnx.Rngs(0)
        agg = lit.LitAggregator(rngs=rngs, dtype=jnp.bfloat16, remat=True)
        pose = lit.LitPoseDecoder(pose_dim=7, rngs=rngs)
        hid = jnp.zeros((18, 2, 776, 2048), jnp.bfloat16)
        roles = jnp.zeros((2, 776), jnp.int32).at[:, :768].set(lit.ROLE_IMAGE).at[:, 768:].set(lit.ROLE_SEMANTIC)
        final, keys, values = agg(hid, *lit.role_masks(roles))
        return final, keys, values, pose(final)

    final, keys, values, pred = nnx.eval_shape(forward)
    assert (final.shape, keys.shape, values.shape, pred.shape) == (
        (2, 100, 768),
        (18, 2, 100, 256),
        (18, 2, 100, 256),
        (2, 7),
    )
    assert final.dtype == keys.dtype == jnp.bfloat16
    assert pred.dtype == jnp.float32
