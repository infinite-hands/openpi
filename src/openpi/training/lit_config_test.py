"""The four LIT rows on the pi05_yam_bagging recipe: they build, they share what the A/B needs them to share, their
freeze filters select what each stage trains (counted at the real dimensions with eval_shape: nothing is allocated),
and the stage-2 row takes its checkpoint from the launcher's flag.

Nothing is trained here and no dataset is read; the data chain runs on the synthetic samples of lit_data_test.
"""

import dataclasses
import sys

import flax.nnx as nnx
import jax
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.shared import nnx_utils
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from openpi.training import lit_data_test as _data
from openpi.training import weight_loaders
from openpi.training.misc import ih_yam_config

BASE = ih_yam_config.LIT_BASE_CONFIG
CONTROL, STAGE1, STAGE2, LITE = (f"{BASE}_{suffix}" for suffix in ("litctl", "lit1", "lit2", "litlite"))
ROWS = (CONTROL, STAGE1, STAGE2, LITE)
REPO_ID = "local/yam_bagging_three"


def _abstract(config: _config.TrainConfig) -> dict:
    """{path: ShapeDtypeStruct-carrying variable} of the real-dimension model, nothing allocated."""
    return nnx.state(nnx.eval_shape(lambda: config.model.create(jax.random.key(0)))).flat_state()


def _count(flat: dict, paths) -> tuple[int, int]:
    """(leaves, parameters) over `paths`."""
    paths = list(paths)
    return len(paths), sum(int(np.prod(flat[p].value.shape)) for p in paths)


def _selected(config: _config.TrainConfig, flat: dict, which: str):
    state = nnx.State.from_flat_path(flat)
    filter_ = config.trainable_filter if which == "trainable" else config.freeze_filter
    return set(state.filter(filter_).flat_state())


# ---- the rows are the recipe's ----


def test_the_rows_exist_under_their_exact_names():
    for name in ROWS:
        assert _config.get_config(name).name == name
    assert ROWS == (
        "pi05_yam_bagging_litctl",
        "pi05_yam_bagging_lit1",
        "pi05_yam_bagging_lit2",
        "pi05_yam_bagging_litlite",
    )


@pytest.mark.parametrize("name", ROWS)
def test_a_row_keeps_the_recipes_dataset_prompt_horizons_and_schedule(name):
    recipe, row = _config.get_config(BASE), _config.get_config(name)
    assert row.data.repo_id == recipe.data.repo_id == REPO_ID
    fields = ("default_prompt", "adapt_to_pi", "use_delta_joint_actions", "repack_transforms", "action_sequence_keys")
    for field in fields:
        assert getattr(row.data, field) == getattr(recipe.data, field), field
    assert row.data.default_prompt == "place one part in the bag"
    for field in ("action_horizon", "action_dim", "max_token_len", "pi05", "dtype", "discrete_state_input"):
        assert getattr(row.model, field) == getattr(recipe.model, field), field
    assert row.model.action_horizon == 30
    fields = ("num_train_steps", "lr_schedule", "optimizer", "save_interval", "keep_period", "seed", "log_interval")
    for field in fields:
        assert getattr(row, field) == getattr(recipe, field), field


def test_the_recipe_itself_is_untouched():
    recipe = _config.get_config(BASE)
    assert recipe.model.lit == "off" and recipe.model.lit_goal_dims == ()
    assert recipe.batch_size == 64 and recipe.ema_decay is None
    assert (recipe.model.paligemma_variant, recipe.model.action_expert_variant) == ("gemma_2b_lora", "gemma_300m_lora")
    assert recipe.weight_loader == weight_loaders.CheckpointWeightLoader(ih_yam_config.PI05_BASE_PARAMS)
    assert recipe.weight_loader.missing_regex == ".*lora.*"


# what each row is: (lit, lora, batch, ema, weight loader)
_TABLE = {
    CONTROL: ("off", False, 32, 0.99, weight_loaders.PaliGemmaWeightLoader()),
    STAGE1: ("stage1", False, 32, None, weight_loaders.PaliGemmaWeightLoader()),
    STAGE2: ("stage2", False, 32, 0.99, weight_loaders.LitStage1WeightLoader()),
    LITE: (
        "stage2",
        True,
        64,
        None,
        weight_loaders.CheckpointWeightLoader(
            ih_yam_config.PI05_BASE_PARAMS, missing_regex=weight_loaders.LIT_MISSING_REGEX
        ),
    ),
}


@pytest.mark.parametrize("name", ROWS)
def test_what_each_row_is(name):
    lit, lora, batch, ema, loader = _TABLE[name]
    row = _config.get_config(name)
    assert row.model.lit == lit
    assert ("lora" in row.model.paligemma_variant, "lora" in row.model.action_expert_variant) == (lora, lora)
    assert (row.batch_size, row.ema_decay) == (batch, ema)
    assert row.weight_loader == loader
    # the freeze filter is the model's own (TrainConfig refuses a LIT row whose filter selects other leaves)
    assert row.freeze_filter == row.model.get_freeze_filter()
    # LIT hyper-parameters are the defaults of the method: 100 latents of 768, 6 groups over the 18 layers
    if lit != "off":
        assert (row.model.lit_num_latents, row.model.lit_dim, row.model.lit_groups) == (100, 768, 6)
        assert row.model.lit_pose_weight == 0.3 and row.model.lit_mask_image and row.model.lit_mask_language
        assert 18 % row.model.lit_groups == 0


def test_only_litlite_opts_in_to_the_lit_missing_regex_and_the_default_stays_lora_only():
    regexes = {name: getattr(_config.get_config(name).weight_loader, "missing_regex", None) for name in ROWS}
    assert regexes == {CONTROL: None, STAGE1: None, STAGE2: None, LITE: weight_loaders.LIT_MISSING_REGEX}
    assert weight_loaders.CheckpointWeightLoader("x").missing_regex == ".*lora.*"
    assert weight_loaders.LIT_MISSING_REGEX == ".*lora.*|lit_.*"
    assert _config.get_config(BASE).weight_loader.missing_regex == ".*lora.*"


# ---- one set of norm statistics ----


def test_the_four_rows_read_one_set_of_statistics_the_production_recipes(tmp_path):
    """Every row pins the assets directory of the run that trained the production bagging checkpoint: no row reads a
    directory another row has to write first, and the control is normalised like the production baseline."""
    production = f"/checkpoints/assets/{ih_yam_config.PART_ADAPT_BASE_ASSETS_DIR.rsplit('/', 1)[-1]}"
    assert production == "/checkpoints/assets/pi05_yam_bagging_three"
    pinned = _config.AssetsConfig(assets_dir=production, asset_id=REPO_ID)
    assert ih_yam_config.LIT_NORM_ASSETS_DIR == production and ih_yam_config.LIT_NORM_ASSET_ID == REPO_ID
    for name in ROWS:
        assert _config.get_config(name).data.assets == pinned, name
        # not any LIT row's own directory (the launcher's statistics pass writes there; a pinned row never reads it)
        assert not _config.get_config(name).data.assets.assets_dir.endswith(tuple(f"/{n}" for n in ROWS))
    # the part-adapt rows pin the same statistics
    assert _config.get_config("pi05_yam_part_adapt").data.assets == pinned

    # resolved through the real data config factory, from a stats file in a stand-in assets directory
    stats = _data._norm_stats()  # noqa: SLF001
    _normalize.save(tmp_path / REPO_ID, stats)
    created = []
    for name in ROWS:
        row = _config.get_config(name)
        data = dataclasses.replace(row.data, assets=_config.AssetsConfig(assets_dir=str(tmp_path), asset_id=REPO_ID))
        created.append(data.create(row.assets_dirs, row.model))
    assert {d.asset_id for d in created} == {REPO_ID}
    for data in created:
        assert set(data.norm_stats) == {"state", "actions"}
        for key in ("state", "actions"):
            np.testing.assert_array_equal(data.norm_stats[key].q01, stats[key].q01)
            np.testing.assert_array_equal(data.norm_stats[key].q99, stats[key].q99)
    # the goal is normalized by the state's statistics at load time, never stored beside them
    assert [bool(d.norm_aliases) for d in created] == [False, True, True, True]


# ---- the driven dimensions ----


def test_lit_goal_dims_are_the_right_arm_and_exclude_the_left_arm_and_its_constant_gripper():
    """The recipe's layout is [L j0..5, L grip, R j0..5, R grip]: the right arm is dims 7-13. The left gripper (dim 6)
    is constant in the corpus and must never be in a pose loss (the evidence is in ih_yam_config's comment)."""
    assert ih_yam_config.LIT_GOAL_DIMS == (7, 8, 9, 10, 11, 12, 13)
    for name in (STAGE1, STAGE2, LITE):
        dims = _config.get_config(name).model.lit_goal_dims
        assert dims == ih_yam_config.LIT_GOAL_DIMS
        assert not set(dims) & set(range(7)), "the left arm and its gripper are out"
        assert max(dims) < 14, "padding dims are never a goal"
        assert 13 in dims, "the right gripper is part of the pose"
    assert _config.get_config(CONTROL).model.lit_goal_dims == ()
    # the grippers are the recipe's absolute (non-delta) dims 6 and 13: the goal keeps its own
    row = _config.get_config(STAGE2)
    data = row.data.create(row.assets_dirs, row.model)
    delta = next(t for t in data.data_transforms.inputs if type(t).__name__ == "DeltaActions")
    assert [i for i, d in enumerate(delta.mask) if not d] == [6, 13]


# ---- what each stage trains, at the real dimensions ----

# Measured with eval_shape on this tree and equal to what the model-stage review expected (lit1 433,910,304 / 27 leaves,
# lit2 3,510,507,799 / 114, litlite 624,030,999 / 114, aggregator 153,659,904, pose decoder 3,414,023).
_EXPECTED = {
    CONTROL: {"leaves": 51, "params": 3_353_433_872, "trainable": (51, 3_353_433_872)},
    STAGE1: {"leaves": 59, "params": 3_357_245_712, "trainable": (27, 433_910_304)},
    STAGE2: {"leaves": 114, "params": 3_510_507_799, "trainable": (114, 3_510_507_799)},
    LITE: {"leaves": 134, "params": 3_560_495_383, "trainable": (114, 624_030_999)},
}
_MODULES = {"lit_aggregator": (55, 153_659_904), "lit_pose_decoder": (8, 3_414_023), "lit_goal_encoder": (8, 3_811_840)}


@pytest.mark.parametrize("name", ROWS)
def test_the_freeze_filter_counts_at_real_dimensions(name):
    row = _config.get_config(name)
    flat = _abstract(row)
    expected = _EXPECTED[name]
    assert _count(flat, flat) == (expected["leaves"], expected["params"])
    trainable = _selected(row, flat, "trainable")
    assert _count(flat, trainable) == expected["trainable"]
    frozen = _selected(row, flat, "frozen")
    assert len(frozen) + len(trainable) == len(flat) and not frozen & trainable
    lit = {p for p in flat if str(p[0]).startswith("lit_")}
    # every lit_* leaf trains, in every stage that has them
    assert lit <= trainable
    for module, counts in _MODULES.items():
        paths = {p for p in lit if p[0] == module}
        assert (_count(flat, paths) if paths else None) == (counts if paths else None), module


def test_the_stages_train_what_the_method_says():
    stage1 = _config.get_config(STAGE1)
    flat = _abstract(stage1)
    trainable = {"/".join(map(str, p)) for p in _selected(stage1, flat, "trainable")}
    frozen = {"/".join(map(str, p)) for p in _selected(stage1, flat, "frozen")}
    # stage 1: the expert (llm *_1, the action/time projections) and the goal encoder; the backbone and SigLIP freeze
    assert any(k.startswith("lit_goal_encoder/") for k in trainable)
    assert all(k.startswith("PaliGemma/img/") or "llm" in k and "_1" not in k for k in frozen)
    assert all(k.startswith(("lit_", "action_", "time_mlp")) or "llm" in k and "_1" in k for k in trainable)

    lite = _config.get_config(LITE)
    flat = _abstract(lite)
    frozen = {"/".join(map(str, p)) for p in _selected(lite, flat, "frozen")}
    trainable = {"/".join(map(str, p)) for p in _selected(lite, flat, "trainable")}
    # litlite: LoRA adapters and everything outside the llm train, the llm's own weights are frozen
    assert frozen and all("llm" in k and "lora" not in k for k in frozen)
    assert any("lora" in k for k in trainable) and any(k.startswith("lit_aggregator/") for k in trainable)

    full = _config.get_config(STAGE2)
    assert not _selected(full, _abstract(full), "frozen"), "the full fine-tune freezes nothing"


def test_a_row_with_the_wrong_freeze_filter_is_refused_at_the_real_dimensions():
    stage1 = _config.get_config(STAGE1)
    with pytest.raises(ValueError, match="freeze_filter"):
        dataclasses.replace(stage1, freeze_filter=nnx.Nothing)
    lite = _config.get_config(LITE)
    with pytest.raises(ValueError, match="freeze_filter"):
        dataclasses.replace(lite, freeze_filter=nnx.Nothing)  # would train the llm's base weights too
    # the same selection written another way is accepted
    same = nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*")))
    dataclasses.replace(lite, freeze_filter=same)


# ---- the data of a row ----


@pytest.mark.parametrize("name", [STAGE1, STAGE2, LITE])
def test_a_rows_data_chain_carries_the_goal_and_every_camera(name, tmp_path):
    """Stage 1 never embeds an image, but the batch has all three cameras: the real row's data config, synthetic
    samples. (lit_data_test runs the same chain on the recipe's data config with the tiny models.)"""
    row = _config.get_config(name)
    data = dataclasses.replace(row.data.create(tmp_path, row.model), norm_stats=_data._norm_stats())  # noqa: SLF001
    assert data.norm_aliases == {"lit_goal": "state"}
    (item,) = _data._transformed(data, [_data._raw_sample(3)])  # noqa: SLF001
    assert set(item["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert all(item["image_mask"].values())
    assert item["lit_goal"].shape == item["state"].shape == (row.model.action_dim,)
    assert item["lit_goal_mask"] == np.bool_(True)


def test_the_control_row_has_no_goal(tmp_path):
    row = _config.get_config(CONTROL)
    data = dataclasses.replace(row.data.create(tmp_path, row.model), norm_stats=_data._norm_stats())  # noqa: SLF001
    assert data.norm_aliases == {}
    (item,) = _data._transformed(data, [_data._raw_sample(3, goal_rows=False)])  # noqa: SLF001
    assert "lit_goal" not in item and "lit_goal_mask" not in item


# ---- the launcher's flags ----


def _cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["train.py", *args])
    return _config.cli()


def test_the_stage2_row_takes_its_stage1_checkpoint_from_the_launchers_flag(monkeypatch):
    parsed = _cli(monkeypatch, STAGE2, "--exp-name", "e", "--weight-loader.params-path", "/ckpt/stage1/params")
    assert parsed.weight_loader == weight_loaders.LitStage1WeightLoader("/ckpt/stage1/params")
    assert parsed.model.lit == "stage2" and parsed.freeze_filter == parsed.model.get_freeze_filter()
    # litlite keeps its opt-in regex when the launcher gives it another warm start
    lite = _cli(monkeypatch, LITE, "--exp-name", "e", "--weight-loader.params-path", "/ckpt/base/params")
    assert lite.weight_loader.params_path == "/ckpt/base/params"
    assert lite.weight_loader.missing_regex == weight_loaders.LIT_MISSING_REGEX
    # the other launcher overrides the recipes use still reach a LIT row
    other = _cli(monkeypatch, STAGE1, "--exp-name", "e", "--batch-size", "8", "--num-train-steps", "5")
    assert (other.batch_size, other.num_train_steps) == (8, 5)


def test_the_hazard_fields_of_other_branches_are_not_set_by_any_row():
    for name in ROWS:
        model = _config.get_config(name).model
        assert isinstance(model, pi0_config.Pi0Config)
        for field, stock in pi0_config._LIT_INCOMPATIBLE_FIELDS:  # noqa: SLF001
            assert getattr(model, field, stock) == stock, (name, field)
