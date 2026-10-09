"""The LIT data pipeline: the goal split, the dataset's delta_timestamps, the forwarding through AlohaInputs, the
normalization alias, the padding, the serve-side stats, and what a stage-1 batch carries.

No model is built here. The real tokenizer (cached under ~/.cache/openpi) and the real transforms of the
pi05_yam_bagging recipe run on synthetic samples shaped like what LeRobotDataset delivers (state (2, 14) at
[t, t + action_horizon] and its `_is_pad` flags, three cameras of torch floats in [0, 1] as (c, h, w)). The query/pad
semantics are the library's own `LeRobotDataset._get_query_indices`, not a copy of them; in an environment where lerobot
is only an import-level stand-in (the offline CPU venv) that is the stand-in's copy of the code, not the pinned
revision's.
"""

import dataclasses
import hashlib
import json

import numpy as np
import pytest
import torch

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.policies import aloha_policy
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

_LeRobotDataset = lerobot_dataset.LeRobotDataset  # the tests below replace the module's attribute with fakes

H = 6  # the action horizon of the synthetic episodes (30 in the real recipe)
STATE_DIM = 14
ACTION_DIM = 32
GOAL_DIMS = tuple(range(7, 14))
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
KEY = _data_loader.LIT_STATE_KEY
EPISODE_ENDS = (10, 25)  # two episodes: frames [0, 10) and [10, 25)
BASE_CONFIG = "pi05_yam_bagging"


def _model_config(lit: str = "off", **overrides) -> pi0_config.Pi0Config:
    kwargs = {"pi05": True, "action_horizon": H, "max_token_len": 200, **overrides}
    if lit != "off":
        kwargs |= {"lit": lit, "lit_goal_dims": GOAL_DIMS, "action_dim": ACTION_DIM}
    return pi0_config.Pi0Config(**kwargs)


def _states_table() -> np.ndarray:
    """One distinct state per frame: frame i is i + 0.01 * dim, so any off-by-one frame is visible."""
    return np.arange(EPISODE_ENDS[-1], dtype=np.float32)[:, None] + 0.01 * np.arange(STATE_DIM, dtype=np.float32)


def _episode_of(frame: int) -> int:
    return 0 if frame < EPISODE_ENDS[0] else 1


def _lerobot_query(frame: int) -> dict:
    """The state fields LeRobotDataset.__getitem__ gives for `frame` when KEY is fetched at [0, H] frames: the
    library's own index/padding code, on a dataset object that has only the episode boundaries."""
    dataset = object.__new__(_LeRobotDataset)
    dataset.episode_data_index = {"from": torch.tensor([0, EPISODE_ENDS[0]]), "to": torch.tensor(EPISODE_ENDS)}
    dataset.delta_indices = {KEY: [0, H]}
    indices, padding = dataset._get_query_indices(frame, _episode_of(frame))  # noqa: SLF001
    return {KEY: torch.as_tensor(_states_table()[indices[KEY]]), **padding}


def _images(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        f"observation.images.{camera}": torch.as_tensor(rng.uniform(size=(3, 24, 32)).astype(np.float32))
        for camera in CAMERAS
    }


def _raw_sample(frame: int, *, split: bool = True, goal_rows: bool = True) -> dict:
    """A sample as the dataset hands it to the transform chain: the LIT dataset after SplitLitGoal (`split`), or the
    stock one (`goal_rows` False: the state at the sample's frame only, no pad flag)."""
    sample = {**_images(frame), "action": torch.as_tensor(np.tile(_states_table()[frame], (H, 1)) + 0.5)}
    if goal_rows:
        sample |= _lerobot_query(frame)
        return _transforms.SplitLitGoal(KEY)(sample) if split else sample
    sample[KEY] = torch.as_tensor(_states_table()[frame])
    return sample


def _norm_stats() -> dict:
    lo, hi = np.full(STATE_DIM, -2.0), np.full(STATE_DIM, 40.0)
    state = _normalize.NormStats(mean=np.zeros(STATE_DIM), std=np.ones(STATE_DIM), q01=lo, q99=hi)
    actions = _normalize.NormStats(mean=np.zeros(STATE_DIM), std=np.ones(STATE_DIM), q01=lo - 1, q99=hi + 1)
    return {"state": state, "actions": actions}


def _quantile(x, stats):
    return (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0


def _data_config(model_config, tmp_path) -> _config.DataConfig:
    """The data config of the pi05_yam_bagging recipe on `model_config`, with synthetic norm stats."""
    factory = _config.get_config(BASE_CONFIG).data
    return dataclasses.replace(factory.create(tmp_path, model_config), norm_stats=_norm_stats())


def _transformed(data_config, samples) -> list:
    """The samples through the real train-time chain (repack, data transforms, Normalize, model transforms)."""
    dataset = _data_loader.transform_dataset(list(samples), data_config)
    return [dataset[i] for i in range(len(dataset))]


# ---- the split ----


@pytest.mark.parametrize("frame", range(EPISODE_ENDS[-1]))
def test_goal_is_the_state_at_t_plus_h_and_the_tail_is_flagged(frame):
    table = _states_table()
    end = EPISODE_ENDS[_episode_of(frame)]
    item = _transforms.SplitLitGoal(KEY)(_raw_sample(frame, split=False))
    np.testing.assert_array_equal(item[KEY], table[frame])
    assert item[KEY].shape == (STATE_DIM,)
    real = frame + H < end
    assert item[_transforms.LIT_GOAL_MASK_KEY] == np.bool_(real)
    assert item[_transforms.LIT_GOAL_MASK_KEY].dtype == np.bool_
    # a clamped goal is the episode's last state (and flagged); a real one is exactly t + H
    np.testing.assert_array_equal(item[_transforms.LIT_GOAL_KEY], table[frame + H if real else end - 1])
    assert f"{KEY}_is_pad" not in item


def test_the_clamped_tail_of_each_episode_is_exactly_the_last_h_frames():
    flagged = [
        frame
        for frame in range(EPISODE_ENDS[-1])
        if not _transforms.SplitLitGoal(KEY)(_raw_sample(frame, split=False))[_transforms.LIT_GOAL_MASK_KEY]
    ]
    expected = [frame for end in EPISODE_ENDS for frame in range(end - H, end)]
    assert flagged == expected


def test_a_sample_that_was_not_fetched_at_two_frames_is_refused():
    with pytest.raises(ValueError, match="fetch it at"):
        _transforms.SplitLitGoal(KEY)(_raw_sample(3, goal_rows=False))
    one_row = {KEY: np.zeros((1, STATE_DIM)), f"{KEY}_is_pad": np.zeros((1,), bool)}
    with pytest.raises(ValueError, match="two frames"):
        _transforms.SplitLitGoal(KEY)(one_row)


# ---- delta_timestamps ----


class _Meta:
    fps = 30
    tasks = {}  # noqa: RUF012

    def __init__(self, repo_id):
        self.repo_id = repo_id


def _capture(monkeypatch, fps=30):
    seen = {}

    class Meta(_Meta):
        pass

    Meta.fps = fps

    class Dataset(list):
        def __init__(self, repo_id, delta_timestamps=None, **kwargs):
            super().__init__()
            seen["delta_timestamps"] = delta_timestamps
            seen["repo_id"] = repo_id

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", Meta)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", Dataset)
    return seen, Dataset


@pytest.mark.parametrize("fps", [30, 50])
def test_lit_off_fetches_exactly_what_the_stock_pipeline_fetched(monkeypatch, tmp_path, fps):
    seen, dataset_type = _capture(monkeypatch, fps)
    config = _model_config()
    data_config = _data_config(config, tmp_path)
    dataset = _data_loader.create_torch_dataset(data_config, H, config)
    # the pin's expression: the action keys at 0..H-1 frames, nothing else
    assert seen["delta_timestamps"] == {key: [t / fps for t in range(H)] for key in data_config.action_sequence_keys}
    assert KEY not in seen["delta_timestamps"]
    # and no transform layer over the dataset
    assert isinstance(dataset, dataset_type)


@pytest.mark.parametrize("fps", [30, 50])
@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_lit_fetches_the_state_now_and_action_horizon_frames_ahead(monkeypatch, tmp_path, fps, stage):
    seen, _ = _capture(monkeypatch, fps)
    config = _model_config(stage)
    data_config = _data_config(config, tmp_path)
    dataset = _data_loader.create_torch_dataset(data_config, H, config)
    stock = {key: [t / fps for t in range(H)] for key in data_config.action_sequence_keys}
    assert seen["delta_timestamps"] == {**stock, KEY: [0.0, H / fps]}
    # at the frame rate the rows land on whole frames: 0 and exactly action_horizon
    assert [round(d * fps) for d in seen["delta_timestamps"][KEY]] == [0, H]
    assert [type(t).__name__ for t in dataset._transform.transforms] == ["SplitLitGoal"]  # noqa: SLF001


def test_create_torch_dataset_hands_the_transform_chain_a_split_sample(monkeypatch, tmp_path):
    """End to end through create_torch_dataset: the dataset's own two-row state comes out as state + goal."""
    config = _model_config("stage2")
    data_config = _data_config(config, tmp_path)

    class Dataset(list):
        def __init__(self, repo_id, delta_timestamps=None, **kwargs):
            super().__init__()

        def __getitem__(self, index):
            return {**_images(index), "action": torch.zeros(H, STATE_DIM), **_lerobot_query(index)}

        def __len__(self):
            return EPISODE_ENDS[-1]

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", _Meta)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", Dataset)
    dataset = _data_loader.create_torch_dataset(data_config, H, config)
    table = _states_table()
    for frame in (0, 3, 8, 12, 24):
        item = dataset[frame]
        np.testing.assert_array_equal(item[KEY], table[frame])
        np.testing.assert_array_equal(
            item[_transforms.LIT_GOAL_KEY], table[min(frame + H, EPISODE_ENDS[_episode_of(frame)] - 1)]
        )
        assert item[_transforms.LIT_GOAL_MASK_KEY] == (frame + H < EPISODE_ENDS[_episode_of(frame)])


def _digest_of_the_stock_pipeline(monkeypatch, tmp_path) -> str:
    """sha256 of what the stock pi05_yam_bagging pipeline fetches and produces on four synthetic samples (every leaf's
    bytes except the images', whose shape and dtype are hashed instead: the resize is the image library's), built
    to be run unchanged against the pin's source tree."""
    h, dim = 30, STATE_DIM
    seen = {}
    config = _config.get_config(BASE_CONFIG)
    data_config = config.data.create(tmp_path, config.model)
    lo, hi = np.linspace(-2.0, -1.0, dim), np.linspace(1.0, 40.0, dim)
    stats = {
        "state": _normalize.NormStats(mean=np.zeros(dim), std=np.ones(dim), q01=lo, q99=hi),
        "actions": _normalize.NormStats(mean=np.zeros(dim), std=np.ones(dim), q01=lo - 1, q99=hi + 1),
    }
    data_config = dataclasses.replace(data_config, norm_stats=stats)

    class Dataset:
        def __init__(self, repo_id, delta_timestamps=None, **kwargs):
            seen["delta_timestamps"] = {key: list(value) for key, value in delta_timestamps.items()}

        def __len__(self):
            return 8

        def __getitem__(self, index):
            rng = np.random.default_rng(index)
            item = {
                f"observation.images.{camera}": torch.as_tensor(rng.uniform(size=(3, 24, 32)).astype(np.float32))
                for camera in CAMERAS
            }
            item["observation.state"] = torch.as_tensor(rng.normal(size=dim).astype(np.float32))
            item["action"] = torch.as_tensor(rng.normal(size=(h, dim)).astype(np.float32))
            return item

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", _Meta)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", Dataset)
    dataset = _data_loader.create_torch_dataset(data_config, h, config.model)
    dataset = _data_loader.transform_dataset(dataset, data_config)
    digest = {}

    def walk(prefix, tree):
        if isinstance(tree, dict):
            for key in sorted(tree):
                walk(f"{prefix}/{key}", tree[key])
            return
        array = np.ascontiguousarray(np.asarray(tree))
        is_image = "/image/" in f"{prefix}/"
        fingerprint = [] if is_image else [hashlib.sha256(array.tobytes()).hexdigest()]
        digest[prefix] = [list(array.shape), str(array.dtype), *fingerprint]

    for index in range(4):
        walk(f"s{index}", dataset[index])
    summary = {"delta_timestamps": seen["delta_timestamps"], "digest": digest}
    return hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()


# Generated by running _digest_of_the_stock_pipeline's body (scratch script stock_dump.py) against the pin's source tree
# (fb9a58b2, extracted with git archive) and against this commit: identical.
_STOCK_PIPELINE_DIGEST = "4d2fed66103a24783678ede575a9f8d8faf1676cdfe9f93e206186550c126c2e"


def test_the_stock_pipeline_is_byte_identical_to_the_pins(monkeypatch, tmp_path):
    assert _digest_of_the_stock_pipeline(monkeypatch, tmp_path) == _STOCK_PIPELINE_DIGEST


# ---- the transform chain ----


def test_aloha_inputs_forwards_the_goal_and_its_mask_and_still_drops_unknown_keys():
    sample = {
        "state": np.arange(STATE_DIM, dtype=np.float32),
        "images": {camera: np.zeros((3, 8, 8), np.float32) for camera in CAMERAS},
        _transforms.LIT_GOAL_KEY: np.arange(STATE_DIM, dtype=np.float32) + 100,
        _transforms.LIT_GOAL_MASK_KEY: np.bool_(True),
        "unknown": np.zeros(3),
    }
    out = aloha_policy.AlohaInputs(adapt_to_pi=False)(sample)
    np.testing.assert_array_equal(out[_transforms.LIT_GOAL_KEY], np.arange(STATE_DIM, dtype=np.float32) + 100)
    assert out[_transforms.LIT_GOAL_MASK_KEY] == np.bool_(True)
    assert "unknown" not in out
    # stock samples have neither key and still get neither
    stock = {k: v for k, v in sample.items() if k not in (_transforms.LIT_GOAL_KEY, _transforms.LIT_GOAL_MASK_KEY)}
    stock_out = aloha_policy.AlohaInputs(adapt_to_pi=False)(stock)
    assert _transforms.LIT_GOAL_KEY not in stock_out and _transforms.LIT_GOAL_MASK_KEY not in stock_out


def test_aloha_inputs_converts_the_goal_like_the_state_when_adapting_to_pi():
    state = np.linspace(0.1, 0.9, STATE_DIM).astype(np.float32)
    sample = {
        "state": state.copy(),
        "images": {"cam_high": np.zeros((3, 8, 8), np.float32)},
        _transforms.LIT_GOAL_KEY: state.copy(),
        _transforms.LIT_GOAL_MASK_KEY: np.bool_(True),
    }
    out = aloha_policy.AlohaInputs(adapt_to_pi=True)(sample)
    np.testing.assert_array_equal(out[_transforms.LIT_GOAL_KEY], out["state"])
    assert not np.array_equal(out["state"], state)  # the conversion did something


def test_carry_lit_goal_adds_the_keys_to_every_repack_and_nothing_else():
    other = _transforms.PadStatesAndActions(8)
    group = _transforms.Group(
        inputs=[_transforms.RepackTransform({"state": "observation.state"}), other],
        outputs=[_transforms.RepackTransform({"x": "y"})],
    )
    carried = _transforms.carry_lit_goal(group)
    assert carried.inputs[0].structure == {
        "state": "observation.state",
        _transforms.LIT_GOAL_KEY: _transforms.LIT_GOAL_KEY,
        _transforms.LIT_GOAL_MASK_KEY: _transforms.LIT_GOAL_MASK_KEY,
    }
    assert carried.inputs[1] is other
    assert carried.outputs == group.outputs
    assert group.inputs[0].structure == {"state": "observation.state"}  # the original is untouched


def test_the_goal_is_normalized_with_the_state_statistics_and_the_mask_is_untouched(tmp_path):
    config = _model_config("stage2")
    data_config = _data_config(config, tmp_path)
    assert data_config.norm_aliases == {_transforms.LIT_GOAL_KEY: "state"}
    # the alias is applied to the data only: the stats that get saved with a checkpoint are the stock ones
    assert set(data_config.norm_stats) == {"state", "actions"}
    stats = data_config.norm_stats
    frame = 3
    (item,) = _transformed(data_config, [_raw_sample(frame)])
    raw = _states_table()[frame + H]
    np.testing.assert_allclose(item[_transforms.LIT_GOAL_KEY][:STATE_DIM], _quantile(raw, stats["state"]), rtol=1e-6)
    assert item[_transforms.LIT_GOAL_MASK_KEY] == np.bool_(True)
    assert item[_transforms.LIT_GOAL_MASK_KEY].dtype == np.bool_
    # at the end of an episode the mask is False and the goal is still the clamped, normalized last state
    (tail,) = _transformed(data_config, [_raw_sample(EPISODE_ENDS[0] - 2)])
    assert tail[_transforms.LIT_GOAL_MASK_KEY] == np.bool_(False)


def test_normalize_aliases_need_the_stats_they_name(tmp_path):
    data_config = dataclasses.replace(
        _data_config(_model_config("stage2"), tmp_path), norm_aliases={_transforms.LIT_GOAL_KEY: "nope"}
    )
    with pytest.raises(ValueError, match="nope"):
        _data_loader.transform_dataset([], data_config)
    # skipping the statistics (compute_norm_stats) skips the alias with them
    _data_loader.transform_dataset([], data_config, skip_norm_stats=True)


def test_state_and_prompt_tokens_never_see_the_goal(tmp_path):
    """The goal is split off before the prompt is tokenized with the discretized state: the tokens of a LIT sample equal
    those of the stock pipeline's sample at the same frame, and the goal does not move them."""
    frame = 3
    lit_config, stock_config = _model_config("stage2"), _model_config()
    (lit,) = _transformed(_data_config(lit_config, tmp_path), [_raw_sample(frame)])
    (stock,) = _transformed(_data_config(stock_config, tmp_path), [_raw_sample(frame, goal_rows=False)])
    np.testing.assert_array_equal(lit["tokenized_prompt"], stock["tokenized_prompt"])
    np.testing.assert_array_equal(lit["tokenized_prompt_mask"], stock["tokenized_prompt_mask"])
    np.testing.assert_array_equal(lit["state"], stock["state"])
    np.testing.assert_array_equal(lit["actions"], stock["actions"])
    # a different goal (a different future), same present: same tokens
    other = _raw_sample(frame)
    other[_transforms.LIT_GOAL_KEY] = other[_transforms.LIT_GOAL_KEY] + 5.0
    (moved,) = _transformed(_data_config(lit_config, tmp_path), [other])
    np.testing.assert_array_equal(moved["tokenized_prompt"], lit["tokenized_prompt"])
    assert not np.array_equal(moved[_transforms.LIT_GOAL_KEY], lit[_transforms.LIT_GOAL_KEY])


def test_a_stock_config_carries_no_lit_key_and_no_alias(tmp_path):
    data_config = _data_config(_model_config(), tmp_path)
    assert data_config.norm_aliases == {}
    (item,) = _transformed(data_config, [_raw_sample(3, goal_rows=False)])
    assert _transforms.LIT_GOAL_KEY not in item and _transforms.LIT_GOAL_MASK_KEY not in item
    for transform in data_config.repack_transforms.inputs:
        assert _transforms.LIT_GOAL_KEY not in transform.structure


def test_pad_states_and_actions_pads_the_goal_like_the_state_and_leaves_the_mask():
    data = {
        "state": np.ones(STATE_DIM, np.float32),
        _transforms.LIT_GOAL_KEY: np.full(STATE_DIM, 2.0, np.float32),
        _transforms.LIT_GOAL_MASK_KEY: np.bool_(True),
        "actions": np.ones((H, STATE_DIM), np.float32),
    }
    out = _transforms.PadStatesAndActions(ACTION_DIM)(data)
    assert out["state"].shape == out[_transforms.LIT_GOAL_KEY].shape == (ACTION_DIM,)
    np.testing.assert_array_equal(out[_transforms.LIT_GOAL_KEY][STATE_DIM:], 0.0)
    np.testing.assert_array_equal(out[_transforms.LIT_GOAL_KEY][:STATE_DIM], 2.0)
    assert out[_transforms.LIT_GOAL_MASK_KEY] == np.bool_(True)
    # without a goal nothing is added
    assert _transforms.LIT_GOAL_KEY not in _transforms.PadStatesAndActions(ACTION_DIM)({"state": np.ones(STATE_DIM)})


def test_output_norm_stats_drops_the_input_only_goal_key():
    """Unnormalize is strict (every stats key must be in the model output) and no output carries the goal."""
    stats = _norm_stats()
    with_goal = {**stats, _transforms.LIT_GOAL_KEY: stats["state"]}
    assert set(_transforms.output_norm_stats(with_goal)) == {"state", "actions"}
    assert _transforms.output_norm_stats(None) is None
    assert _transforms.output_norm_stats(stats) == stats
    outputs = {"actions": np.zeros((H, STATE_DIM)), "state": np.zeros(STATE_DIM)}
    with pytest.raises(ValueError, match=_transforms.LIT_GOAL_KEY):  # the bug it prevents
        _transforms.Unnormalize(with_goal, use_quantiles=True)(dict(outputs))
    _transforms.Unnormalize(_transforms.output_norm_stats(with_goal), use_quantiles=True)(dict(outputs))


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_a_lit_batch_through_the_real_transform_chain_has_every_camera(tmp_path, stage):
    """Stage 1 never embeds an image, but compute_loss still needs the images dict (preprocess_observation raises on a
    missing key): the batch the loader hands the model carries all three cameras with their masks, and the goal."""
    data_config = _data_config(_model_config(stage), tmp_path)
    items = _transformed(data_config, [_raw_sample(frame) for frame in (0, 4, 9, 12)])
    batch = _data_loader._collate_fn(items)  # noqa: SLF001
    observation = _model.Observation.from_dict(batch)
    assert set(observation.images) == set(_model.IMAGE_KEYS)
    assert all(observation.image_masks[key].all() for key in _model.IMAGE_KEYS)
    assert observation.images["base_0_rgb"].shape == (4, 224, 224, 3)
    assert observation.lit_goal.shape == (4, ACTION_DIM)
    assert observation.lit_goal_mask.shape == (4,) and observation.lit_goal_mask.dtype == np.bool_
    assert observation.lit_goal_mask.tolist() == [True, False, False, True]
    assert observation.state.shape == (4, ACTION_DIM)
    assert batch["actions"].shape == (4, H, ACTION_DIM)


def test_lit_needs_the_lerobot_loader(tmp_path):
    model = _model_config("stage2")
    config = dataclasses.replace(_config.get_config(BASE_CONFIG), model=model, freeze_filter=model.get_freeze_filter())
    data_config = dataclasses.replace(_data_config(model, tmp_path), rlds_data_dir="/nowhere")

    class Factory(_config.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            return data_config

    with pytest.raises(ValueError, match="LeRobot data loader"):
        _data_loader.create_data_loader(dataclasses.replace(config, data=Factory(repo_id="x")))
