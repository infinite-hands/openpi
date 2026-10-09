"""Weight loading for LIT: the opt-in missing regex, the stage-1 -> stage-2 handoff and its fingerprint, the refusal to
load a LIT checkpoint into a stock (lit="off") model at both layers that load params (Pi0Config.load, what serving
uses, and the trainer's CheckpointWeightLoader), and the orbax round trip of a stage-2 train state with its EMA.

Parameter trees are the tiny dummy variants' (lit_test_utils) with random values, saved and restored through orbax like
a real checkpoint. The image tower is the stub encoder (one Dense layer) because only the tree matters here; the real
dimensions are only ever built with eval_shape.
"""

import dataclasses
import functools
import logging
import os
import re
from unittest import mock

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import numpy as np
import orbax.checkpoint as ocp
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.models import lit_test_utils as _utils
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import siglip as _siglip
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import sharding
from openpi.training import weight_loaders

from . import lit_train_test as _train_test
from . import train


_REAL_SIGLIP = _siglip.Module  # captured before any fixture swaps in the stub


@pytest.fixture(autouse=True)
def stub():
    with _utils.stub_image_encoder():
        yield


def _shapes(stage) -> dict:
    """The pure dict of ShapeDtypeStruct params of a stage's tiny model: what the trainer hands a weight loader."""
    config = _train_test._model_config(stage)
    return nnx.state(nnx.eval_shape(lambda: config.create(jax.random.key(0)))).to_pure_dict()


def _checkpoint(tmp_path, stage, seed=0) -> tuple[str, dict]:
    """(params path, the params) of a random tiny checkpoint of a stage, written through orbax as the trainer does."""
    rng = np.random.default_rng(seed)
    params = jax.tree.map(lambda s: rng.normal(size=s.shape).astype(s.dtype), _shapes(stage))
    path = tmp_path / f"{stage}_{seed}" / "params"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(path, {"params": params})
    return str(path), params


def _flat(tree) -> dict:
    return traverse_util.flatten_dict(tree, sep="/")


# (checkpoint stage, target stage, loads)
_DIRECTIONS = [
    ("stage2", "off", False),
    ("stage1", "off", False),
    ("off", "stage2", False),
    ("stage1", "stage2", False),  # without the opt-in regex
    ("stage2", "stage1", False),
    ("off", "off", True),
    ("stage1", "stage1", True),
    ("stage2", "stage2", True),
]


@pytest.mark.parametrize(("source", "target", "loads"), _DIRECTIONS)
def test_model_load_refuses_a_lit_checkpoint_in_a_stock_model_and_every_other_mismatch(
    tmp_path, source, target, loads
):
    """Pi0Config.load (serving): drops extras, and used to drop a LIT checkpoint's lit_* leaves without a word."""
    _, params = _checkpoint(tmp_path, source)
    config = _train_test._model_config(target)
    if loads:
        assert config.load(params).lit == target
        return
    with pytest.raises(ValueError) as error:
        config.load(params)
    if target == "off":
        assert "LIT" in str(error.value) and "lit_" in str(error.value)


@pytest.mark.parametrize(("source", "target", "loads"), _DIRECTIONS)
def test_the_trainer_loader_refuses_a_lit_checkpoint_in_a_stock_model_and_every_other_mismatch(
    tmp_path, source, target, loads
):
    """The trainer path: CheckpointWeightLoader + the pytree equality train.py demands."""
    path, _ = _checkpoint(tmp_path, source)
    loader = weight_loaders.CheckpointWeightLoader(path)
    shapes = _shapes(target)
    if loads:
        loaded = train._load_weights_and_validate(loader, shapes)
        assert set(_flat(loaded)) == set(_flat(shapes))
        return
    with pytest.raises(ValueError) as error:
        train._load_weights_and_validate(loader, shapes)
    if target == "off":
        # the loader's own refusal, before any merge: not a leftover structure error
        assert "LIT" in str(error.value) and "lit_" in str(error.value) and "silently" in str(error.value)


def test_the_opt_in_regex_lets_a_stage2_config_take_a_checkpoint_without_its_lit_modules(tmp_path):
    shapes = _shapes("stage2")
    for source in ("off", "stage1"):  # a pi05_base-like stock tree, and a stage-1 checkpoint (goal encoder dropped)
        path, params = _checkpoint(tmp_path, source)
        loader = weight_loaders.CheckpointWeightLoader(path, missing_regex=weight_loaders.LIT_MISSING_REGEX)
        loaded = _flat(train._load_weights_and_validate(loader, shapes))
        assert set(loaded) == {k for k in _flat(shapes) if not k.startswith("lit_")}
        for key, value in loaded.items():
            np.testing.assert_array_equal(value, _flat(params)[key])
    # the default stays LoRA-only, and the field is a plain dataclass field (the CLI can reach it)
    assert weight_loaders.CheckpointWeightLoader("x").missing_regex == ".*lora.*"
    assert weight_loaders.CheckpointWeightLoader("x") == weight_loaders.CheckpointWeightLoader("x", ".*lora.*")


def test_the_lit_missing_regex_is_anchored():
    pattern = re.compile(weight_loaders.LIT_MISSING_REGEX)
    assert pattern.fullmatch("lit_aggregator/groups/to_key/kernel")
    assert pattern.fullmatch("lit_pose_decoder/fc1/bias")
    assert pattern.fullmatch("PaliGemma/llm/layers/attn/q_einsum/lora_a")
    # a path that merely contains "lit_" is not a LIT module: an unanchored `.*lit_.*` would tolerate its absence
    assert not pattern.fullmatch("PaliGemma/llm/layers/split_heads/kernel")
    assert re.fullmatch(".*lit_.*", "PaliGemma/llm/layers/split_heads/kernel")


def test_no_stock_parameter_path_matches_the_lit_missing_regex_at_real_dimensions():
    """Abstract (eval_shape): every leaf of the stock model is either a LoRA leaf or required."""
    pattern = re.compile(weight_loaders.LIT_MISSING_REGEX)

    def paths(config):
        with mock.patch.object(_siglip, "Module", _REAL_SIGLIP):  # the real image tower, not the stub
            state = nnx.state(nnx.eval_shape(lambda: config.create(jax.random.key(0))))
        return ["/".join(map(str, path)) for path in state.flat_state()]

    stock = paths(pi0_config.Pi0Config(pi05=True, action_horizon=30))
    assert any(p.startswith("PaliGemma/img/") and not p.endswith("head/kernel") for p in stock), "real SigLIP"
    assert not [p for p in stock if pattern.fullmatch(p)]
    lora = paths(
        pi0_config.Pi0Config(
            pi05=True, action_horizon=30, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        )
    )
    matching = [p for p in lora if pattern.fullmatch(p)]
    assert matching and all("lora" in p for p in matching)


# ---- the stage-1 handoff ----


def test_the_handoff_loader_keeps_every_backbone_and_expert_leaf_and_drops_the_goal_encoder(tmp_path):
    path, params = _checkpoint(tmp_path, "stage1")
    shapes = _shapes("stage2")
    loaded = _flat(train._load_weights_and_validate(weight_loaders.LitStage1WeightLoader(path), shapes))
    checkpoint = _flat(params)
    assert set(loaded) == {k for k in checkpoint if not k.startswith("lit_")} == {
        k for k in _flat(shapes) if not k.startswith("lit_")
    }
    assert any(k.startswith("lit_goal_encoder/") for k in checkpoint)
    assert not any(k.startswith("lit_") for k in loaded), "the stage-2 lit_* modules start fresh; no goal encoder"
    for key, value in loaded.items():
        np.testing.assert_array_equal(value, checkpoint[key], err_msg=key)


def test_the_handoff_loader_without_a_path_says_what_to_pass():
    with pytest.raises(ValueError, match="--weight-loader.params-path"):
        weight_loaders.LitStage1WeightLoader().load(_shapes("stage2"))


@pytest.mark.parametrize("source", ["off", "stage2"])
def test_the_handoff_loader_refuses_a_checkpoint_that_is_not_stage1(tmp_path, source):
    path, _ = _checkpoint(tmp_path, source)
    with pytest.raises(ValueError, match="not a LIT stage-1 checkpoint"):
        weight_loaders.LitStage1WeightLoader(path).load(_shapes("stage2"))


@pytest.mark.parametrize("target", ["off", "stage1"])
def test_the_handoff_loader_refuses_a_target_that_is_not_stage2(tmp_path, target):
    path, _ = _checkpoint(tmp_path, "stage1")
    with pytest.raises(ValueError, match="lit='stage2'"):
        weight_loaders.LitStage1WeightLoader(path).load(_shapes(target))


def _expert_and_backbone_keys(tree):
    keys = [k for k in _flat(tree) if k.startswith("PaliGemma/llm/")]
    return next(k for k in keys if "_1" in k), next(k for k in keys if "_1" not in k)


def test_the_fingerprint_catches_a_merge_that_does_not_hand_off_what_it_should(tmp_path, monkeypatch):
    """The loader checks its result instead of trusting the merge: corrupt the merge four ways, each must raise."""
    path, params = _checkpoint(tmp_path, "stage1")
    shapes = _shapes("stage2")
    loader = weight_loaders.LitStage1WeightLoader(path)
    assert loader.load(shapes)  # the honest merge passes
    expert, backbone = _expert_and_backbone_keys(shapes)
    real_merge = weight_loaders._merge_params  # noqa: SLF001

    def corrupted(change):
        def merge(loaded, reference, *, missing_regex):
            flat = _flat(real_merge(loaded, reference, missing_regex=missing_regex))
            change(flat)
            return traverse_util.unflatten_dict(flat, sep="/")

        return merge

    def perturb(flat):
        flat[backbone] = flat[backbone] + 1.0

    def fresh_expert(flat):
        flat[expert] = _flat(shapes)[expert]  # the expert left at the model's own initialisation

    def loaded_lit(flat):
        key = next(k for k in _flat(shapes) if k.startswith("lit_aggregator/"))
        flat[key] = np.zeros(_flat(shapes)[key].shape, np.float32)

    def goal_encoder_survives(flat):
        flat["lit_goal_encoder/fc1/kernel"] = np.zeros((2, 2), np.float32)

    for change, match in [
        (perturb, "not the stage-1 checkpoint's value"),
        (fresh_expert, "not the stage-1 checkpoint's value"),
        (loaded_lit, "must start fresh"),
        (goal_encoder_survives, "survived"),
    ]:
        monkeypatch.setattr(weight_loaders, "_merge_params", corrupted(change))
        with pytest.raises(ValueError, match=match):
            loader.load(shapes)
    monkeypatch.setattr(weight_loaders, "_merge_params", real_merge)

    # a checkpoint that lacks a backbone leaf: neither loaded nor LoRA, so the handoff is incomplete, said here and not
    # left to the trainer's later structure error
    incomplete = {k: v for k, v in _flat(params).items() if k != backbone}
    short_path = tmp_path / "short" / "params"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(short_path, {"params": traverse_util.unflatten_dict(incomplete, sep="/")})
    with pytest.raises(ValueError, match="incomplete"):
        weight_loaders.LitStage1WeightLoader(str(short_path)).load(shapes)


# ---- through the trainer: a real stage-1 run, then a stage-2 train state initialised from it ----


class _Loader(_train_test._Loader):
    pass


def _patched_main(monkeypatch, batch):
    monkeypatch.setattr(_data_loader, "create_data_loader", lambda *a, **k: _Loader(batch))
    monkeypatch.setattr(train, "init_logging", lambda: None)


def _leaves(state) -> dict:
    return {k: np.asarray(v) for k, v in _train_test._flat(state).items()}


def test_a_stage1_run_hands_off_to_a_stage2_train_state(tmp_path, monkeypatch):
    stage1_model = _train_test._model_config("stage1")
    stage1 = _train_test._train_config(
        stage1_model, checkpoint_base_dir=str(tmp_path), exp_name="stage1", num_train_steps=2
    )
    _patched_main(monkeypatch, _train_test._batch(stage1_model))
    train.main(stage1)
    saved = stage1.checkpoint_dir / "1" / "params"
    assert saved.exists()
    checkpoint = _flat(_model.restore_params(saved, restore_type=np.ndarray))
    assert any(k.startswith("lit_goal_encoder/") for k in checkpoint)

    stage2_model = _train_test._model_config("stage2")
    stage2 = _train_test._train_config(
        stage2_model, ema_decay=0.99, weight_loader=weight_loaders.LitStage1WeightLoader(str(saved))
    )
    init_rng = jax.random.key(7)
    state, _ = train.init_train_state(stage2, init_rng, sharding.make_mesh(1), resume=False)
    params = _leaves(state.params)
    assert not any(k.startswith("lit_goal_encoder/") for k in params)

    # the stage-2 train state is the model's own initialisation (the rng init_train_state gives it) with the stage-1
    # checkpoint's backbone and expert in it
    fresh = _leaves(nnx.state(stage2_model.create(jax.random.split(init_rng)[1])))
    assert sorted(params) == sorted(fresh)
    loaded = fresh_lit = 0
    for key, value in params.items():
        if key.startswith("lit_"):
            np.testing.assert_array_equal(value, fresh[key], err_msg=f"{key} is not the fresh initialisation")
            fresh_lit += 1
        else:
            np.testing.assert_array_equal(value, checkpoint[key].astype(value.dtype), err_msg=key)
            loaded += 1
    assert loaded and fresh_lit
    for key, value in _leaves(state.ema_params).items():
        np.testing.assert_array_equal(value, params[key], err_msg=f"ema {key}")

    # and it trains: one stage-2 step is finite, with a real pose loss and a gradient in every lit_* module
    step = jax.jit(functools.partial(train.train_step, stage2))
    new_state, info = step(jax.random.key(0), state, _train_test._batch(stage2_model))
    assert np.isfinite(float(info["loss"])) and float(info["pose_loss"]) > 0.0
    assert float(info["grad_norm_lit_aggregator"]) > 0.0 and float(info["grad_norm_lit_pose_decoder"]) > 0.0
    assert int(new_state.step) == 1


# ---- orbax round trip and resume ----


def test_a_stage2_train_state_with_ema_survives_an_orbax_round_trip(tmp_path):
    model = _train_test._model_config("stage2")
    config = _train_test._train_config(model, ema_decay=0.99, checkpoint_base_dir=str(tmp_path), exp_name="rt")
    state = _train_test._state(config)
    step = _train_test._jitted(config)
    for i in range(2):
        state, _ = step(jax.random.key(i), state, _train_test._batch(model, i))
    manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=None, overwrite=False, resume=False
    )
    assert not resuming
    _checkpoints.save_state(manager, state, _Loader(None), 2)
    manager.wait_until_finished()

    abstract, _ = train.init_train_state(config, jax.random.key(0), sharding.make_mesh(1), resume=True)
    restored = _checkpoints.restore_state(manager, abstract, _Loader(None))
    assert int(restored.step) == int(state.step) == 2
    for name in ("params", "ema_params"):
        want, got = _leaves(getattr(state, name)), _leaves(getattr(restored, name))
        assert sorted(want) == sorted(got)
        assert any(k.startswith("lit_aggregator/") for k in got) and any(k.startswith("lit_pose_decoder/") for k in got)
        for key in want:
            np.testing.assert_array_equal(got[key], want[key], err_msg=f"{name}/{key}")
    for want, got in zip(jax.tree.leaves(state.opt_state), jax.tree.leaves(restored.opt_state), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    # the `params` item a serve or a stage hand-off reads is the EMA, lit_* included
    saved = _flat(_model.restore_params(config.checkpoint_dir / "2" / "params", restore_type=np.ndarray))
    for key, value in _leaves(state.ema_params).items():
        np.testing.assert_array_equal(saved[key], value, err_msg=key)


def test_a_stage2_run_resumes_where_it_stopped(tmp_path, monkeypatch, caplog):
    model = _train_test._model_config("stage2")
    _patched_main(monkeypatch, _train_test._batch(model))
    caplog.set_level(logging.INFO)
    config = _train_test._train_config(
        model, ema_decay=0.99, checkpoint_base_dir=str(tmp_path), exp_name="resume", num_train_steps=2
    )
    train.main(config)
    first = _train_test._step_lines(caplog.text)
    assert [p["step"] for p in first] == [0, 1]
    caplog.clear()
    train.main(dataclasses.replace(config, resume=True, num_train_steps=4))
    second = _train_test._step_lines(caplog.text)
    assert [p["step"] for p in second] == [2, 3]
    assert all(p["pose_loss"] > 0.0 for p in second)
    saved = _flat(_model.restore_params(config.checkpoint_dir / "3" / "params", restore_type=np.ndarray))
    assert any(k.startswith("lit_aggregator/") for k in saved)


def test_the_stage2_row_loader_takes_its_checkpoint_from_the_command_line():
    """The loader is built from the config with no path and gets it at launch, as the launcher passes it for every
    config: --weight-loader.params-path."""
    import tyro

    default = weight_loaders.LitStage1WeightLoader()
    config = _config.TrainConfig(name="x", weight_loader=default)
    parsed = tyro.cli(
        _config.TrainConfig, default=config, args=["--exp-name", "e", "--weight-loader.params-path", "/some/params"]
    )
    assert parsed.weight_loader == weight_loaders.LitStage1WeightLoader("/some/params")
