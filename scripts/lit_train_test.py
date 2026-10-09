"""The LIT trainer: the pose loss enters the step once, the Step line carries the parts, the stage-2 guard, the EMA of
frozen leaves, and the TrainConfig freeze-filter check.

Tiny dummy variants (lit_test_utils), float32, CPU. The subject here is the step and its plumbing, not the image tower,
so the models are built with the stub image encoder (lit_test_utils.stub_image_encoder: one Dense over the patches);
nothing about the lit=off numerics depends on it, and the stock-step comparison below runs the same stub on both sides.
A freshly created pi0.5 is prefix-blind (zero-initialised adaRMS gates), so every model the gradient tests use goes
through randomize_zero_init.
"""

import dataclasses
import functools
import logging
import os
import re

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.models import lit_test_utils as _utils
from openpi.models import model as _model
from openpi.shared import nnx_utils
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import lit_train as _lit_train
from openpi.training import optimizer as _optimizer
from openpi.training import sharding
from openpi.training import utils as training_utils

from . import train

BATCH = _utils.BATCH
SCHEDULE = _optimizer.CosineDecaySchedule(warmup_steps=1, peak_lr=1e-2, decay_steps=10, decay_lr=1e-3)


@pytest.fixture(autouse=True)
def stub():
    with _utils.stub_image_encoder():
        yield


@pytest.fixture(autouse=True)
def cold_jax_cache(tmp_path, monkeypatch):
    """No persistent compilation cache for the test: train.main points jax at a cache directory (here a fresh one), and
    on CPU (jax 0.5.3) an executable of `init_train_state` or of the train step that is read back from it returns
    different values for the same program (a warm start loaded through a weight loader came back as garbage, and a
    train step segfaulted), donated buffers included. The cache is switched off, so nothing is written or read."""
    monkeypatch.setenv("OPENPI_JAX_COMPILATION_CACHE_DIR", str(tmp_path / "jax-cache"))
    previous = jax.config.jax_enable_compilation_cache
    jax.config.update("jax_enable_compilation_cache", False)
    yield
    jax.config.update("jax_enable_compilation_cache", previous)


def _model_config(stage, **overrides):
    return _utils.make_tiny_config(lit=stage, **overrides)


def _train_config(model, **overrides) -> _config.TrainConfig:
    fields = {
        "name": "lit_train_test",
        "model": model,
        "data": _config.FakeDataConfig(),
        "batch_size": BATCH,
        "lr_schedule": SCHEDULE,
        "freeze_filter": model.get_freeze_filter(),
        "ema_decay": None,
        "wandb_enabled": False,
        "num_train_steps": 2,
        "log_interval": 1,
        "save_interval": 1,
        "keep_period": None,
        "num_workers": 0,
        **overrides,
    }
    return _config.TrainConfig(**fields)


def _batch(model, seed=0, *, goal_mask=(True, True)):
    obs = _utils.make_observation(seed, config=model)
    if model.lit != "off":
        goal = np.zeros((BATCH, model.action_dim), np.float32)
        goal[:, : _utils.ROBOT_DIMS] = np.random.default_rng(seed + 1).normal(size=(BATCH, _utils.ROBOT_DIMS))
        obs = obs.replace(lit_goal=jnp.asarray(goal), lit_goal_mask=jnp.asarray(goal_mask, jnp.bool_))
    return obs, _utils.make_actions(seed, config=model)


def _state(config: _config.TrainConfig, *, randomize=True, seed=0) -> training_utils.TrainState:
    """init_train_state's own recipe on a model built here, so its zero-initialised leaves can be randomized."""
    model = _utils.build_model(config.model, seed)
    if randomize:
        model = _utils.randomize_zero_init(model, seed)
    params = nnx.state(model)
    params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    return training_utils.TrainState(
        step=0,
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(params.filter(config.trainable_filter)),
        ema_decay=config.ema_decay,
        ema_params=None if config.ema_decay is None else params,
    )


def _flat(state) -> dict:
    return {"/".join(map(str, path)): np.asarray(var.value) for path, var in state.flat_state().items()}


def _pins_train_step(config, rng, state, batch):
    """scripts/train.py's train_step as the pin had it (fb9a58b2), verbatim apart from the names it is reached by."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model, rng, observation, actions):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@functools.cache
def _jitted(config: _config.TrainConfig, *, stock: bool = False):
    return jax.jit(functools.partial(_pins_train_step if stock else train.train_step, config))


def _assert_states_equal(a: training_utils.TrainState, b: training_utils.TrainState):
    assert int(a.step) == int(b.step)
    for name in ("params", "ema_params"):
        left, right = getattr(a, name), getattr(b, name)
        assert (left is None) == (right is None)
        if left is not None:
            x, y = _flat(left), _flat(right)
            assert sorted(x) == sorted(y)
            for key in x:
                np.testing.assert_array_equal(x[key], y[key], err_msg=f"{name}/{key}")
    for x, y in zip(jax.tree.leaves(a.opt_state), jax.tree.leaves(b.opt_state), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


# ---- lit=off is the pin's step ----


@pytest.mark.parametrize(
    ("freeze", "ema_decay"),
    [
        pytest.param("nothing", 0.99, id="full fine-tune with EMA"),
        pytest.param("backbone", None, id="frozen backbone without EMA"),
    ],
)
def test_the_stock_step_is_bit_identical_to_the_pins(freeze, ema_decay):
    """lit=off: the has_aux plumbing and the EMA helper change neither the update, the EMA, the optimizer state nor the
    three logged scalars, over two steps. (Full fine-tune freezes nothing and LoRA configs run without EMA: the stock
    configs the EMA change cannot touch.)"""
    model = _model_config("off")
    freeze_filter = nnx.Nothing if freeze == "nothing" else nnx_utils.PathRegex(".*llm.*")
    config = _train_config(model, freeze_filter=freeze_filter, ema_decay=ema_decay)
    new, old = _state(config), _state(config)
    batch = _batch(model)
    for step in range(2):
        rng = jax.random.key(step)
        new, new_info = _jitted(config)(rng, new, batch)
        old, old_info = _jitted(config, stock=True)(rng, old, batch)
        assert set(new_info) == set(old_info) == {"loss", "grad_norm", "param_norm"}
        for key in old_info:
            assert float(new_info[key]) == float(old_info[key]), key
        _assert_states_equal(new, old)
    assert float(old_info["grad_norm"]) > 0.0


# ---- the pose loss enters once ----


def _total_and_grads(model, params_state, weight, observation, actions, rng):
    """(loss_with_parts value, its gradient over every param) with model.lit_pose_weight set to `weight`."""
    model.lit_pose_weight = weight

    def fn(params):
        merged = nnx.merge(nnx.graphdef(model), params)
        return _lit_train.loss_with_parts(merged, rng, observation, actions)

    (loss, parts), grads = jax.value_and_grad(fn, has_aux=True)(params_state)
    return float(loss), parts, grads


def test_the_pose_weight_is_applied_exactly_once():
    model_config = _model_config("stage2")
    model = _utils.randomize_zero_init(_utils.build_model(model_config))
    params = nnx.state(model)
    observation, actions = _batch(model_config)
    rng = jax.random.key(3)
    action_loss, aux = model.compute_loss_and_aux(rng, observation, actions, train=True)
    action_loss = float(jnp.mean(action_loss))
    pose = float(aux["pose_loss"])
    assert pose > 0.0

    totals, grads = {}, {}
    for weight in (0.0, 0.3, 0.6):
        totals[weight], parts, grads[weight] = _total_and_grads(model, params, weight, observation, actions, rng)
        assert float(parts["action_loss"]) == pytest.approx(action_loss, rel=1e-6)
        assert float(parts["pose_loss"]) == pytest.approx(pose, rel=1e-6)
    # value: action + w * pose, once (twice would be action + 2 w pose)
    for weight, total in totals.items():
        assert total == pytest.approx(action_loss + weight * pose, rel=1e-5)
    # gradient: linear in w with the same slope at both steps, and the w=0 gradient carries no pose term at all
    lit_keys = [k for k in _flat(grads[0.0]) if k.startswith("lit_pose_decoder/")]
    assert lit_keys
    flat = {w: _flat(g) for w, g in grads.items()}
    for key in lit_keys:
        assert not flat[0.0][key].any(), "the pose decoder gets no gradient from the action loss"
        assert flat[0.3][key].any()
    for key, base in flat[0.0].items():
        half, full = flat[0.3][key] - base, flat[0.6][key] - base
        np.testing.assert_allclose(full, 2.0 * half, rtol=1e-4, atol=1e-6, err_msg=key)


def test_a_stage2_step_without_a_pose_loss_is_refused():
    model_config = _model_config("stage2")
    model = _utils.randomize_zero_init(_utils.build_model(model_config))
    observation, actions = _batch(model_config)
    real = model.compute_loss_and_aux

    def without_pose(*args, **kwargs):
        loss, aux = real(*args, **kwargs)
        return loss, {k: v for k, v in aux.items() if k != "pose_loss"}

    model.compute_loss_and_aux = without_pose
    with pytest.raises(ValueError, match="pose path is not wired"):
        _lit_train.loss_with_parts(model, jax.random.key(0), observation, actions)
    # the train step reaches the same guard
    config = _train_config(model_config)
    state = _state(config)
    state = dataclasses.replace(state, model_def=nnx.graphdef(model))
    with pytest.raises(ValueError, match="pose path is not wired"):
        train.train_step(config, jax.random.key(0), state, (observation, actions))


# ---- what the Step line carries ----

_STAGE1_KEYS = {
    "loss",
    "grad_norm",
    "param_norm",
    "action_loss",
    "pose_copy_baseline",
    "grad_norm_expert",
    "grad_norm_lit_goal_encoder",
}
_STAGE2_KEYS = {
    "loss",
    "grad_norm",
    "param_norm",
    "action_loss",
    "pose_loss",
    "pose_copy_baseline",
    "grad_norm_backbone",
    "grad_norm_expert",
    "grad_norm_lit_aggregator",
    "grad_norm_lit_pose_decoder",
}


@pytest.mark.parametrize(("stage", "keys"), [("stage1", _STAGE1_KEYS), ("stage2", _STAGE2_KEYS)])
def test_the_step_reports_the_parts_and_the_module_gradient_norms(stage, keys):
    model = _model_config(stage)
    config = _train_config(model)
    state = _state(config)
    state, info = _jitted(config)(jax.random.key(0), state, _batch(model))
    assert set(info) == keys
    assert all(np.isfinite(float(v)) for v in info.values())
    for key in keys - {"pose_copy_baseline"}:
        assert float(info[key]) > 0.0, key
    # the total is the action loss plus the weighted pose loss, once (stage 1 has no pose loss)
    pose = float(info["pose_loss"]) if "pose_loss" in info else 0.0
    expected = float(info["action_loss"]) + model.lit_pose_weight * pose
    assert float(info["loss"]) == pytest.approx(expected, rel=1e-5)
    # numeric k=v, as the Step line prints them
    assert all(re.fullmatch(r"-?\d+\.\d{4}", f"{float(v):.4f}") for v in info.values())


def test_stage1_trains_the_expert_and_the_goal_encoder_only():
    """The backbone's gradient norm is not reported in stage 1 because its leaves are frozen, not trainable."""
    model = _model_config("stage1")
    config = _train_config(model)
    trainable = {"/".join(map(str, p)) for p in _state(config).params.filter(config.trainable_filter).flat_state()}
    assert any(k.startswith("lit_goal_encoder/") for k in trainable)
    backbone = [k for k in trainable if k.startswith("PaliGemma/img/") or re.fullmatch(r"PaliGemma/llm/(?!.*_1).*", k)]
    assert not backbone


def test_module_grad_norms_partition_every_parameter_of_the_model():
    """Every trainable leaf lands in exactly one module and the norms recombine to the global gradient norm."""
    model = _model_config("stage2")
    config = _train_config(model)
    state = _state(config)
    net = nnx.merge(state.model_def, state.params)
    observation, actions = _batch(model)
    grads = nnx.grad(
        lambda m: _lit_train.loss_with_parts(m, jax.random.key(0), observation, actions)[0],
        argnums=nnx.DiffState(0, config.trainable_filter),
    )(net)
    norms = _lit_train.module_grad_norms(grads)
    assert set(norms) == {f"grad_norm_{m}" for m in ("backbone", "expert", "lit_aggregator", "lit_pose_decoder")}
    recombined = float(jnp.sqrt(sum(jnp.square(v) for v in norms.values())))
    assert recombined == pytest.approx(float(optax.global_norm(grads)), rel=1e-5)


# ---- the EMA of frozen leaves ----


def test_ema_leaves_a_frozen_leaf_alone_where_the_plain_average_shrinks_it():
    """Stage 1 freezes the backbone (bf16) and runs here with an EMA: the EMA a checkpoint saves as `params` keeps the
    frozen leaves bit for bit, while averaging a bf16 leaf against itself loses a little every step."""
    model = _model_config("stage1")
    config = _train_config(model, ema_decay=0.99)
    state = _state(config)
    frozen = set(state.params.filter(config.freeze_filter).flat_state())
    trainable = set(state.params.filter(config.trainable_filter).flat_state())
    assert frozen and trainable
    start = _flat(state.ema_params)
    old_formula = jax.tree.map(lambda a, b: 0.99 * a + 0.01 * b, state.ema_params, state.params)
    step = _jitted(config)
    for i in range(3):
        state, _ = step(jax.random.key(i), state, _batch(model, i))
    ema, params = _flat(state.ema_params), _flat(state.params)
    for path in frozen:
        key = "/".join(map(str, path))
        np.testing.assert_array_equal(ema[key], start[key], err_msg=key)
        np.testing.assert_array_equal(ema[key], params[key], err_msg=key)
    moved = [k for path in trainable if not np.array_equal(ema[k := "/".join(map(str, path))], start[k])]
    assert moved, "the trainable leaves are still averaged"
    # what the plain formula does to the same frozen leaf (bf16 0.99 * x + 0.01 * x is not x)
    shrunk = [
        k
        for path in frozen
        if not np.array_equal(_flat(old_formula)[k := "/".join(map(str, path))], start[k])
    ]
    assert shrunk, "the plain average changes a frozen bf16 leaf: this is what ema_update prevents"


def test_ema_update_averages_every_leaf_exactly_as_before_when_nothing_is_frozen():
    ema = nnx.state(nnx.Linear(3, 2, rngs=nnx.Rngs(0)))
    new = jax.tree.map(lambda x: x + 1.0, ema)
    got = training_utils.ema_update(ema, new, 0.99, nnx.Param)
    expected = jax.tree.map(lambda old, n: 0.99 * old + (1 - 0.99) * n, ema, new)
    assert jax.tree.structure(got) == jax.tree.structure(expected)
    for x, y in zip(jax.tree.leaves(got), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


# ---- TrainConfig checks the freeze filter of a LIT model ----


def test_a_stage1_config_left_at_the_default_freeze_filter_is_refused():
    model = _model_config("stage1")
    with pytest.raises(ValueError, match="freeze_filter must select the same leaves") as error:
        _config.TrainConfig(name="x", model=model)  # freeze_filter defaults to nnx.Nothing: trains every leaf
    assert "model.get_freeze_filter()" in str(error.value)


def test_the_freeze_filter_check_compares_selections_not_objects():
    model = _model_config("stage1")
    expected = model.get_freeze_filter()
    # the same leaves chosen another way: accepted, and the same filter built again is accepted without any model
    same_selection = nnx.All(
        nnx.Any(nnx_utils.PathRegex("PaliGemma/llm/.*"), nnx_utils.PathRegex("PaliGemma/img/.*")),
        nnx.Not(nnx_utils.PathRegex(".*_1.*")),
    )
    _config.TrainConfig(name="x", model=model, freeze_filter=same_selection)
    _config.TrainConfig(name="x", model=model, freeze_filter=model.get_freeze_filter())
    assert expected == model.get_freeze_filter()
    # one leaf more frozen: refused, with the leaves it differs on
    with pytest.raises(ValueError, match="differ on"):
        more = nnx.Any(expected, nnx_utils.PathRegex(".*action_in_proj.*"))
        _config.TrainConfig(name="x", model=model, freeze_filter=more)


def test_a_stage2_config_with_the_wrong_freeze_filter_is_refused_and_lit_off_is_not_checked():
    stage2 = _model_config("stage2")
    _config.TrainConfig(name="x", model=stage2, freeze_filter=stage2.get_freeze_filter())
    with pytest.raises(ValueError, match="freeze_filter"):
        _config.TrainConfig(name="x", model=stage2, freeze_filter=nnx_utils.PathRegex("lit_.*"))
    # the check is for LIT models: a stock config keeps whatever filter it has
    _config.TrainConfig(name="x", model=_model_config("off"), freeze_filter=nnx_utils.PathRegex("lit_.*"))


# ---- the loop: the Step line reaches the log ----


class _Loader:
    def __init__(self, batch):
        self._batch = batch

    def data_config(self):
        return _config.DataConfig()

    def __iter__(self):
        while True:
            yield self._batch


def _run_main(tmp_path, monkeypatch, caplog, model_config, **overrides):
    batch = _batch(model_config)
    monkeypatch.setattr(_data_loader, "create_data_loader", lambda *a, **k: _Loader(batch))
    monkeypatch.setattr(train, "init_logging", lambda: None)
    config = _train_config(model_config, checkpoint_base_dir=str(tmp_path), exp_name="run", **overrides)
    caplog.set_level(logging.INFO)
    train.main(config)
    return config


def _step_lines(text) -> list[dict]:
    lines = []
    for match in re.finditer(r"Step (\d+): ([^\n(]*)", text):
        values = dict(pair.split("=") for pair in match.group(2).strip().split(", "))
        lines.append({"step": int(match.group(1)), **{k: float(v) for k, v in values.items()}})
    return lines


@pytest.mark.parametrize(("stage", "keys"), [("stage1", _STAGE1_KEYS), ("stage2", _STAGE2_KEYS)])
def test_the_step_line_reaches_the_log_with_every_curve(tmp_path, monkeypatch, caplog, stage, keys):
    """train.main's Step line through the logging module (the pin already logs it there beside pbar.write; nothing was
    added): points > 0, all numeric, each step once."""
    _run_main(tmp_path, monkeypatch, caplog, _model_config(stage), num_train_steps=3)
    points = _step_lines(caplog.text)
    assert len(points) >= 3, caplog.text[-2000:]
    # one line per step through logging, as at the pin: a repeated step would be a duplicate point in a metrics tap
    steps = [p["step"] for p in points]
    assert steps == sorted(set(steps)), steps
    for point in points:
        assert keys <= set(point) - {"step"}
        assert all(np.isfinite(v) for v in point.values())
    assert all(p["action_loss"] > 0 for p in points)
    if stage == "stage2":
        assert all(p["pose_loss"] > 0 for p in points)


def test_a_stock_run_logs_no_lit_keys(tmp_path, monkeypatch, caplog):
    _run_main(tmp_path, monkeypatch, caplog, _model_config("off"), num_train_steps=2)
    points = _step_lines(caplog.text)
    assert points
    for point in points:
        assert {"loss", "grad_norm", "param_norm"} <= set(point)
        assert not any(k.startswith(("pose", "action_loss", "grad_norm_")) for k in point)


# ---- scripts that do not train LIT ----


def test_the_pytorch_trainer_refuses_a_lit_config():
    from . import train_pytorch

    for stage in ("stage1", "stage2"):
        with pytest.raises(ValueError, match="no LIT path"):
            train_pytorch.train_loop(_train_config(_model_config(stage)))


def test_the_train_script_calls_the_lit_loss_helper_and_nothing_else_builds_a_loss():
    """Source text, so that a merge cannot take the stage-2 pose-loss guard away unseen: ih/wam-aux-loss,
    ih/wam-aux-frozen-diag and ih/wam-future-tokens (openpi) rewrite train.py's loss_fn and EMA, and git merges them
    cleanly. The spec names scripts/train.py and scripts/train_multi_node.py; this fork has no multi-node trainer."""
    scripts = os.path.dirname(os.path.abspath(__file__))
    assert not os.path.exists(os.path.join(scripts, "train_multi_node.py"))
    text = open(os.path.join(scripts, "train.py")).read()
    assert "return _lit_train.loss_with_parts(model, rng, observation, actions)" in text
    assert "nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)" in text
    assert "**_lit_train.lit_info(model, parts, grads)" in text
    assert "training_utils.ema_update(" in text
    assert "model.compute_loss(" not in text, "the stock loss call must stay behind loss_with_parts"
    # the pin's logging, unchanged: one pbar.write and one logging.info of the Step line, never a second logging call
    assert text.count("pbar.write(") == 1 and text.count('logging.info(f"Step') == 1
    # the helper is where the guard lives, and it is the only caller of a model's loss outside tests
    helper = open(os.path.join(os.path.dirname(_lit_train.__file__), "lit_train.py")).read()
    assert '"pose_loss" not in aux' in helper and "model.lit_pose_weight * aux[" in helper
    sources = {
        name: open(os.path.join(scripts, name)).read()
        for name in sorted(os.listdir(scripts))
        if name.endswith(".py") and not name.endswith("_test.py")
    }
    callers = [name for name, source in sources.items() if ".compute_loss(" in source]
    assert callers == []
    # the one other trainer cannot train LIT at all
    assert "no LIT path" in open(os.path.join(scripts, "train_pytorch.py")).read()
