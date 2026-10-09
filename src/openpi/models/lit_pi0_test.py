"""The LIT path of Pi0: config, goal field, role layout, two-pass loss, masks, gradients, freeze filter, sampling.

Everything runs on the tiny harness of lit_test_utils (dummy Gemma variants, depth 4, real SigLIP unless the opt-in stub
encoder is on, float32, batch 2, three cameras of 256 tokens plus 8 prompt tokens: one prefix of 776 tokens; openpi has no
history blocks and no memory dict). Real dimensions are only ever built with `nnx.eval_shape`. A freshly created pi0.5 is
prefix-blind and its SigLIP head is zero, so every model here goes through randomize_zero_init.

Which tests are exact and which are tolerances. Comparisons against the lit_golden fixtures are bit-exact and only valid
on the toolchain that made them (and with the real SigLIP): they carry `exact_equality` and SKIP with a "not run: ..."
reason anywhere else. Every other test is platform-independent: exact-equality of two computations on the same machine
(a masked column contributes exactly zero), or a tolerance measured on this CPU and recorded next to it.

What the tolerances were measured as, on this CPU in float32 with the real SigLIP (not invented): the two-pass loss
against the stock joint-pass loss differs by a relative 5.0e-8 to 3.0e-7 over rng seeds 0-2 and train False/True; the jitted
loss against the eager one by 2.0e-7; a mathematically inert parameter (the aggregator's key bias, over which softmax is
invariant) moves the loss by 2.0e-7. _REL = 1e-5 leaves more than 30x over each. The hard-mask invariances are exactly 0.0,
and each comes with a power probe: started from the raw valid-token count the same inputs move the velocity by 0.17 to
0.76 (its scale is 3.1). Sampling: the eager training velocity and the eager sampling-cache velocity are exactly equal
(0.0) in float32 and bfloat16; through the compiled sampling loop they differ by 7.6e-8 (float32) and 2.3e-3 (bfloat16: one
bf16 rounding step is 3.9e-3) relative; the jitted sampler against the eager one by 1.3e-7.

Run the whole file with `-s` to see the measured numbers and the real-dimension parameter counts it prints.
"""
# ruff: noqa: SLF001

import contextlib
import dataclasses
import functools
import json

import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import lit as _lit
from openpi.models import lit_test_utils as _utils
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config
from openpi.models.lit_golden import gen_baseline as _gen
from openpi.training import weight_loaders

_NOT_RUN = _gen.fixture_unavailable_reason()
exact_equality = pytest.mark.skipif(_NOT_RUN is not None, reason=_NOT_RUN or "")

_REL = 1e-5
B = _utils.BATCH
TEXT = _utils.MAX_TOKEN_LEN
IMAGE_TOKENS = 256
CAMERAS = 3
PREFIX_LEN = CAMERAS * IMAGE_TOKENS + TEXT
# Valid prompt tokens of the two samples (make_observation: max_token_len - 3 * i) and the camera of sample 1 that is
# missing (DEFAULT_MASKED: right wrist).
TEXT_LENGTHS = (TEXT, TEXT - 3)
ACTION_DIM = _utils.ACTION_DIM
GOAL_DIMS = tuple(range(_utils.ROBOT_DIMS))
RNG = jax.random.key(0)


def _goal(seed: int = 0):
    goal = np.zeros((B, ACTION_DIM), np.float32)
    goal[:, : len(GOAL_DIMS)] = np.random.default_rng(seed).normal(size=(B, len(GOAL_DIMS)))
    return jnp.asarray(goal)


def _obs(seed: int = 0, *, goal_seed: int = 0, goal_mask=(True, True), config=None):
    return _utils.make_observation(
        seed, config=config, lit_goal=_goal(goal_seed), lit_goal_mask=jnp.asarray(goal_mask, jnp.bool_)
    )


def _pre(obs):
    return _model.preprocess_observation(None, obs, train=False)


def _path(path) -> str:
    return "/".join(str(p) for p in path)


def _flat(state) -> dict:
    return {_path(path): np.asarray(var.value) for path, var in state.flat_state().items()}


def _is_lit_path(path: str) -> bool:
    return path.startswith("lit_")


# ---- fixtures: one raw and one randomized model per stage, shared by the whole file ----


@functools.cache
def _stage_model(stage: str, **overrides):
    """(raw manifest, randomized model): the raw manifest is taken before randomize_zero_init."""
    config = _utils.make_tiny_config(lit=stage, **overrides)
    raw = _utils.build_model(config)
    manifest = _utils.param_manifest(raw)
    return manifest, _utils.randomize_zero_init(raw)


@pytest.fixture(scope="module")
def stage2():
    return _stage_model("stage2")[1]


@pytest.fixture(scope="module")
def stage1():
    return _stage_model("stage1")[1]


@contextlib.contextmanager
def _switches(model, *, image: bool, language: bool):
    """The two stage-2 masks set to `image` / `language` for the block, restored to the default (both on) after."""
    model.lit_mask_image, model.lit_mask_language = image, language
    try:
        yield model
    finally:
        model.lit_mask_image = model.lit_mask_language = True


@functools.cache
def _prefix(stage: str, obs_key: str):
    """The LIT prefix pass of a stage's model over a named observation (cached: each one runs SigLIP eagerly)."""
    return _stage_model(stage)[1]._lit_prefix_pass(_pre(_OBS[obs_key]))


_BASE = _obs(0)
_IMAGES_ONLY = _BASE.replace(images=_utils.make_observation(7).images)
_TEXT_ONLY = _BASE.replace(tokenized_prompt=_utils.make_observation(7).tokenized_prompt)
_STATE_ONLY = _BASE.replace(state=_utils.make_observation(7).state)
_GOAL_ONLY = _BASE.replace(lit_goal=_goal(3))
_OBS = {"base": _BASE, "images": _IMAGES_ONLY, "text": _TEXT_ONLY, "state": _STATE_ONLY, "goal": _GOAL_ONLY}


def _x_t_and_time(seed: int = 0):
    config = _utils.make_tiny_config()
    return _utils.fixed_noise(seed, config=config), jnp.full((B,), 0.6, jnp.float32)


def _velocity(stage: str, prefix, obs_key: str = "base", extra=None):
    """The action expert's velocity over `prefix`, with its own appended columns or the given `extra`."""
    model = _stage_model(stage)[1]
    x_t, time = _x_t_and_time()
    if extra is None:
        extra = prefix.extra
    return model._suffix_velocity(_pre(_OBS[obs_key]), x_t, time, prefix.kv_cache, prefix.visible, prefix.offset, extra)


def _equal(a, b):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _differs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


# ---- config, spec and Observation plumbing (no model) ----

_LIT_FIELDS = {
    "lit_num_latents",
    "lit_dim",
    "lit_groups",
    "lit_heads",
    "lit_kv_dim",
    "lit_pose_tokens",
    "lit_goal_tokens",
    "lit_pose_weight",
    "lit_mask_image",
    "lit_mask_language",
    "lit_goal_dims",
}


def test_config_defaults_are_the_stock_model():
    config = pi0_config.Pi0Config()
    assert config.lit == "off"
    assert (config.lit_num_latents, config.lit_dim, config.lit_groups, config.lit_heads) == (100, 768, 6, 8)
    assert (config.lit_kv_dim, config.lit_pose_tokens, config.lit_goal_tokens) == (256, 8, 8)
    assert (config.lit_pose_weight, config.lit_mask_image, config.lit_mask_language) == (0.3, True, True)
    assert config.lit_goal_dims == ()
    assert {f.name for f in dataclasses.fields(config)} >= _LIT_FIELDS | {"lit"}
    assert config.get_freeze_filter() is nnx.Nothing


def _tiny(**overrides):
    return _utils.make_tiny_config(**{"lit": "stage2", **overrides})


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"lit": "stage3"}, "lit must be"),
        ({"lit_goal_dims": ()}, "lit_goal_dims"),
        ({"lit_goal_dims": (0, 0)}, "distinct"),
        ({"lit_goal_dims": (0, ACTION_DIM)}, "distinct"),
        ({"lit_groups": 3}, "divide the backbone depth"),
        ({"lit_kv_dim": 32}, "num_kv_heads"),
        ({"lit_heads": 5}, "divisible"),
        ({"lit_pose_tokens": 7}, "lit_pose_tokens"),
        ({"lit_num_latents": 0}, "lit_num_latents"),
        ({"lit_pose_weight": -0.1}, "lit_pose_weight"),
        ({"pi05": False}, "pi05"),
    ],
)
def test_config_validation(overrides, match):
    with pytest.raises(ValueError, match=match):
        _tiny(**overrides)


def test_config_validation_is_skipped_for_lit_off():
    # nothing about lit_* is read when the feature is off, so a stock config with junk lit_* values still builds
    pi0_config.Pi0Config(lit_groups=5, lit_goal_dims=(99,))


def test_stage1_refuses_lora_variants():
    kwargs = {"pi05": True, "paligemma_variant": "gemma_2b_lora", "lit_goal_dims": GOAL_DIMS, "action_dim": 14}
    with pytest.raises(ValueError, match="no LoRA"):
        pi0_config.Pi0Config(lit="stage1", **kwargs)
    pi0_config.Pi0Config(lit="stage2", **kwargs)


@dataclasses.dataclass(frozen=True)
class _WithOtherForkFields(pi0_config.Pi0Config):
    """The fields other branches add to Pi0Config, spelled and defaulted as the branches have them (the merge hazards
    in its comment: openpi ih/wam-future-tokens, ih/per-dim-loss-weighting; StreamPI ih/spatial-forcing, ih/vlash,
    ih/race). No field is invented here: every one is on a real branch."""

    future_tokens: str = "off"
    action_dim_weights: tuple[float, ...] | None = None
    spatial_layer: int | None = None
    vlash_branches: int = 0
    state_cond: bool = False
    image_keys: tuple[str, ...] = _model.IMAGE_KEYS
    race: bool = False


_HAZARDS = {
    "future_tokens": "visible",
    "action_dim_weights": (1.0,) * ACTION_DIM,
    "spatial_layer": 3,
    "vlash_branches": 2,
    "state_cond": True,
    "image_keys": ("base_0_rgb", "left_wrist_0_rgb"),
    "race": True,
}


def test_the_guarded_names_are_the_ones_the_pi0_config_comment_lists():
    assert {name for name, _ in pi0_config._LIT_INCOMPATIBLE_FIELDS} == set(_HAZARDS)  # noqa: SLF001
    assert not any(f.name == "encode_only_active_cameras" for f in dataclasses.fields(pi0_config.Pi0Config))


@pytest.mark.parametrize("name", sorted(_HAZARDS))
def test_merge_hazards_are_refused(name):
    kwargs = {"pi05": True, "lit_goal_dims": GOAL_DIMS, "action_dim": ACTION_DIM}
    _WithOtherForkFields(lit="stage2", **kwargs)  # every stock value is accepted
    _WithOtherForkFields(lit="stage2", image_keys=list(_model.IMAGE_KEYS), **kwargs)  # a list of the same keys too
    for stage in ("stage1", "stage2"):
        with pytest.raises(ValueError, match=name):
            _WithOtherForkFields(lit=stage, **kwargs, **{name: _HAZARDS[name]})
    _WithOtherForkFields(**{name: _HAZARDS[name]})  # lit off: untouched


def test_real_dimension_configs_are_valid():
    for stage in ("stage1", "stage2"):
        pi0_config.Pi0Config(lit=stage, pi05=True, action_horizon=30, lit_goal_dims=GOAL_DIMS, max_token_len=200)


def test_inputs_spec_carries_the_goal_only_for_lit():
    for stage in ("stage1", "stage2"):
        config = _utils.make_tiny_config(lit=stage)
        spec, _ = config.inputs_spec(batch_size=3)
        assert spec.lit_goal.shape == (3, ACTION_DIM)
        assert spec.lit_goal.dtype == jnp.float32
        assert spec.lit_goal_mask.shape == (3,)
        assert spec.lit_goal_mask.dtype == jnp.bool_
        fake = config.fake_obs(batch_size=3)
        assert fake.lit_goal.shape == (3, ACTION_DIM)
        assert bool(jnp.all(fake.lit_goal_mask))
    stock, _ = _utils.make_tiny_config().inputs_spec(batch_size=3)
    assert stock.lit_goal is None
    assert stock.lit_goal_mask is None


def test_observation_carries_the_goal_through_from_dict_to_dict_and_preprocess():
    obs = _obs(0, goal_mask=(True, False))
    data = obs.to_dict()
    assert sorted(k for k in data if k.startswith("lit_")) == ["lit_goal", "lit_goal_mask"]
    _equal(data["lit_goal"], obs.lit_goal)
    again = _model.Observation.from_dict(data)
    _equal(again.lit_goal, obs.lit_goal)
    _equal(again.lit_goal_mask, obs.lit_goal_mask)
    processed = _pre(obs)
    _equal(processed.lit_goal, obs.lit_goal)
    _equal(processed.lit_goal_mask, obs.lit_goal_mask)
    _equal(processed.state, obs.state)
    # a stock observation: no goal, and its dict round trip (None-valued lit keys included) still loads
    plain = _model.Observation.from_dict({k: v for k, v in data.items() if not k.startswith("lit_")})
    assert plain.lit_goal is None
    assert plain.lit_goal_mask is None
    assert _pre(plain).lit_goal is None
    round_trip = _model.Observation.from_dict(plain.to_dict())
    assert round_trip.lit_goal is None
    assert round_trip.lit_goal_mask is None


def test_a_goal_and_its_mask_must_come_together_in_from_dict():
    data = _obs(0).to_dict()
    for dropped in ("lit_goal_mask", "lit_goal"):
        with pytest.raises(ValueError, match="lit_goal and lit_goal_mask"):
            _model.Observation.from_dict({k: v for k, v in data.items() if k != dropped})
        with pytest.raises(ValueError, match="lit_goal and lit_goal_mask"):
            _model.Observation.from_dict({**data, dropped: None})


def test_goal_must_be_state_wide(stage2):
    narrow = _BASE.replace(lit_goal=_BASE.lit_goal[:, : len(GOAL_DIMS)])
    with pytest.raises(ValueError, match="wide"):
        stage2.compute_loss_and_aux(RNG, narrow, _utils.make_actions(1))


# ---- the checkpoint directions: a LIT checkpoint never loads silently as a stock model ----


def _abstract_params(stage: str):
    """The pure-dict params of a model of this stage, as ShapeDtypeStructs (nothing is allocated)."""
    config = _utils.make_tiny_config(lit=stage)
    return config, nnx.state(nnx.eval_shape(config.create, jax.random.key(0))).to_pure_dict()


def test_lit_off_refuses_a_lit_checkpoint_and_the_other_directions_raise():
    configs = {stage: _abstract_params(stage) for stage in ("off", "stage1", "stage2")}
    configs["off"][0].load(configs["off"][1])  # a stock checkpoint loads into a stock config
    for source in ("stage1", "stage2"):
        with pytest.raises(ValueError, match="cannot load a LIT checkpoint"):
            configs["off"][0].load(configs[source][1])  # not silently as a stock model
    for source, target in (("off", "stage2"), ("stage1", "stage2"), ("stage2", "stage1"), ("off", "stage1")):
        with pytest.raises(ValueError, match="different structure"):
            configs[target][0].load(configs[source][1])
    configs["stage2"][0].load(configs["stage2"][1])
    configs["stage1"][0].load(configs["stage1"][1])


# ---- lit=off is the stock model ----


@exact_equality
def test_lit_off_equals_the_baseline_fixture_and_never_enters_the_lit_path(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the LIT loss ran for lit='off'")

    monkeypatch.setattr(_pi0.Pi0, "_compute_loss_lit", refuse)
    model = _gen.get_model("float32", "rand")
    config = _utils.make_tiny_config()
    observation = _utils.make_observation(_gen.OBSERVATION_SEEDS[0], config=config)
    actions = _utils.make_actions(_gen.ACTIONS_SEED, config=config)
    assert model.lit == "off"
    assert not any(_is_lit_path(p) for p in _flat(nnx.state(model)))
    with np.load(_gen.FIXTURE_DIR / "baseline_loss.npz") as fixture:
        for seed, train in _gen.LOSS_RUNS["f32_rand_eager"]:
            want = fixture[_gen.loss_key("f32_rand_eager", seed, train=train)]
            loss, aux = model.compute_loss_and_aux(jax.random.key(seed), observation, actions, train=train)
            assert loss.dtype == want.dtype
            _equal(loss, want)
            _equal(model.compute_loss(jax.random.key(seed), observation, actions, train=train), want)
            assert aux == {}


@functools.cache
def _off_manifest():
    return _utils.param_manifest(_utils.build_model(_utils.make_tiny_config()))


def test_lit_modules_come_last_so_the_stock_parameters_and_rng_stream_do_not_move():
    """A lit=off model and the non-lit_* part of a stage model, built in this process, are identical leaf for leaf
    (same type, shape, dtype, values): creating the lit_* modules last moves nothing."""
    off = _off_manifest()
    for stage in ("stage1", "stage2"):
        manifest = _stage_model(stage)[0]
        base = {p: v for p, v in manifest.items() if not _is_lit_path(p)}
        assert base == off, stage
        assert {p.split("/")[0] for p in manifest if p not in off} <= {
            "lit_goal_encoder",
            "lit_aggregator",
            "lit_pose_decoder",
        }


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_lit_off_param_tree_structure_is_the_fixture(dtype):
    expected = json.loads((_gen.FIXTURE_DIR / "baseline_params.json").read_text())[dtype]
    got = _utils.param_manifest(_gen.get_model(dtype, "raw"))
    if _utils.stub_image_encoder_enabled():
        expected = {p: v for p, v in expected.items() if not p.startswith("PaliGemma/img/")}
        got = {p: v for p, v in got.items() if not p.startswith("PaliGemma/img/")}
    assert sorted(got) == sorted(expected)
    assert {p: (v["type"], v["shape"], v["dtype"]) for p, v in got.items()} == {
        p: (v["type"], v["shape"], v["dtype"]) for p, v in expected.items()
    }


@exact_equality
def test_lit_off_param_values_are_the_fixture_and_lit_modules_come_last():
    expected = json.loads((_gen.FIXTURE_DIR / "baseline_params.json").read_text())["float32"]
    for stage in ("stage1", "stage2"):
        manifest = _stage_model(stage)[0]
        base = {p: v for p, v in manifest.items() if not _is_lit_path(p)}
        assert sorted(base) == sorted(expected), stage
        moved = [p for p in expected if base[p]["sha256"] != expected[p]["sha256"]]
        assert not moved, f"adding the lit_* modules moved the init of {moved}"


# ---- the prefix layout: roles and visibility ----


def test_prefix_roles_follow_the_layout_and_the_masks(stage2):
    obs = _pre(_BASE)
    tokens, prefix_mask, ar_mask = stage2.embed_prefix(obs)
    assert tokens.shape[1] == PREFIX_LEN
    assert not bool(jnp.any(ar_mask)), "openpi's prefix is one block: every token shares the same attention group"
    roles = np.asarray(stage2.prefix_roles(obs, PREFIX_LEN))
    assert roles.shape == (B, PREFIX_LEN)
    # a mask-free reading of the layout: [cam 0 | cam 1 | cam 2 | text]
    expected = np.zeros_like(roles)
    for sample in range(B):
        for camera, name in enumerate(obs.images):
            if bool(obs.image_masks[name][sample]):
                start = camera * IMAGE_TOKENS
                expected[sample, start : start + IMAGE_TOKENS] = _lit.ROLE_IMAGE
        start = CAMERAS * IMAGE_TOKENS
        expected[sample, start : start + TEXT_LENGTHS[sample]] = _lit.ROLE_SEMANTIC
    np.testing.assert_array_equal(roles, expected)
    # and the roles are valid exactly where embed_prefix says a token is
    np.testing.assert_array_equal(roles != _lit.ROLE_PAD, np.asarray(prefix_mask))
    assert (roles[1] == _lit.ROLE_IMAGE).sum() == 2 * IMAGE_TOKENS  # sample 1 lost a camera
    assert (roles[0] == _lit.ROLE_IMAGE).sum() == 3 * IMAGE_TOKENS


def test_prefix_roles_rejects_a_layout_that_does_not_fit(stage2):
    with pytest.raises(ValueError, match="do not fit"):
        stage2.prefix_roles(_pre(_BASE), PREFIX_LEN + 1)


def test_stage1_prefix_is_the_language_only(stage1, stage2):
    obs = _pre(_BASE)
    tokens, mask, ar_mask = stage1.embed_prefix_text(obs)
    assert tokens.shape == (B, TEXT, 64)
    assert not bool(jnp.any(ar_mask))
    _equal(mask, obs.tokenized_prompt_mask)
    full, *_ = stage2.embed_prefix(obs)
    _equal(tokens, full[:, CAMERAS * IMAGE_TOKENS :])  # the same embedding as the stock text tokens
    roles = np.asarray(stage1.prefix_roles(obs, TEXT))
    np.testing.assert_array_equal(roles, np.where(np.asarray(obs.tokenized_prompt_mask), _lit.ROLE_SEMANTIC, 0))


def _columns(visible, sample):
    return np.flatnonzero(np.asarray(visible[sample]))


def _expected_columns(sample: int, which: str) -> list[int]:
    language = [CAMERAS * IMAGE_TOKENS + j for j in range(TEXT_LENGTHS[sample])]
    images = [
        c * IMAGE_TOKENS + j
        for c in range(CAMERAS)
        for j in range(IMAGE_TOKENS)
        if not (sample == 1 and c == 2)  # the camera sample 1 lost
    ]
    return {"language": language, "images": images, "all": sorted(language + images)}[which]


def test_stage2_visibility_is_per_role(stage2):
    both = _prefix("stage2", "base")
    assert not bool(jnp.any(both.visible)), "image and language/state columns are hidden"
    assert both.extra[2].shape == (B, 6)
    assert bool(jnp.all(both.extra[2]))
    obs = _pre(_BASE)
    for mask_image, mask_language, want in ((True, False, "language"), (False, True, "images"), (False, False, "all")):
        with _switches(stage2, image=mask_image, language=mask_language):
            prefix = stage2._lit_prefix_pass(obs)
        for sample in range(B):
            np.testing.assert_array_equal(_columns(prefix.visible, sample), _expected_columns(sample, want))


def test_latent_columns_have_the_cache_layout_and_dtype(stage2):
    prefix = _prefix("stage2", "base")
    layers, _, s, kv_heads, head_dim = prefix.kv_cache[0].shape
    assert (layers, s, kv_heads, head_dim) == (4, PREFIX_LEN, 1, 16)
    for part in prefix.extra[:2]:
        assert part.shape == (layers, B, 6, kv_heads, head_dim)
    assert prefix.latents.shape == (B, 6, 32)
    # layer l's latent K/V come from group l // 2 of the aggregator: the two layers of a group differ, the groups too
    keys = np.asarray(prefix.extra[0])
    assert not np.array_equal(keys[0], keys[1])
    np.testing.assert_array_equal(np.asarray(prefix.count), [PREFIX_LEN, PREFIX_LEN - IMAGE_TOKENS - 3])


# ---- the two-pass loss against the stock joint pass ----


def _two_pass_loss(model, rng, observation, actions, *, train, hidden_extra=False):
    """The stock loss recomputed from the pieces: prefix pass, then the suffix over [cache; (no) extra; own block]."""
    pre_rng = jax.random.split(rng, 3)[0]
    observation = _model.preprocess_observation(pre_rng, observation, train=train)
    noise, time = _utils.training_noise_and_time(rng, actions.shape[0])
    x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
    prefix = model._lit_prefix_pass(observation)
    extra = None
    if hidden_extra:
        extra = (prefix.extra[0], prefix.extra[1], jnp.zeros_like(prefix.extra[2]))
    v_t = model._suffix_velocity(observation, x_t, time, prefix.kv_cache, prefix.visible, prefix.offset, extra)
    return jnp.mean(jnp.square(v_t - (noise - actions)), axis=-1)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("train", [False, True])
def test_two_pass_with_no_lit_mask_and_no_latents_equals_the_stock_joint_pass(stage2, seed, train):
    """With the masks off and the latent columns absent (or hidden), two passes are the stock joint pass: this checks
    the suffix positions and the mask layout against the stock code on the same weights."""
    rng, obs, actions = jax.random.key(seed), _utils.make_observation(0), _utils.make_actions(1)
    with _switches(stage2, image=False, language=False):
        stage2.lit = "off"  # runs the unmodified stock body on the very same weights
        try:
            stock = stage2.compute_loss(rng, obs, actions, train=train)
        finally:
            stage2.lit = "stage2"
        two_pass = _two_pass_loss(stage2, rng, obs, actions, train=train)
        hidden = _two_pass_loss(stage2, rng, obs, actions, train=train, hidden_extra=True)
    scale = float(jnp.max(jnp.abs(stock)))
    assert _differs(two_pass, stock) <= _REL * scale
    assert _differs(hidden, stock) <= _REL * scale
    print(f"two-pass vs stock, seed {seed} train {train}: {_differs(two_pass, stock) / scale:.2e} relative")


# ---- the hard mask ----


def test_hard_mask_makes_the_velocity_independent_of_images_language_and_state():
    """Latents held fixed (the ones of the base observation), every other input changes: the action rows can see
    nothing but those latents and their own block, so the velocity is bit-for-bit the same."""
    reference_prefix = _prefix("stage2", "base")
    reference = _velocity("stage2", reference_prefix)
    for name in ("images", "text", "state", "goal"):
        prefix = _prefix("stage2", name)
        # the perturbation really changed what the backbone computed (the state and the goal are not in the prefix)
        changed = not np.array_equal(np.asarray(prefix.kv_cache[0]), np.asarray(reference_prefix.kv_cache[0]))
        assert changed == (name in ("images", "text")), name
        _equal(_velocity("stage2", prefix, name, extra=reference_prefix.extra), reference)


def test_with_live_latents_the_velocity_depends_on_images_and_on_language():
    reference = _velocity("stage2", _prefix("stage2", "base"))
    control = _velocity("stage2", _prefix("stage2", "base"))
    assert _differs(control, reference) == 0.0, "the computation is deterministic, so any difference below is real"
    on_images = _differs(_velocity("stage2", _prefix("stage2", "images"), "images"), reference)
    on_text = _differs(_velocity("stage2", _prefix("stage2", "text"), "text"), reference)
    print(
        f"live latents: velocity moves by {on_images:.3e} (images) and {on_text:.3e} (language), scale "
        f"{float(np.max(np.abs(reference))):.3e}"
    )
    assert on_images > 0.0
    assert on_text > 0.0


def test_without_the_mask_the_prefix_columns_are_read(stage2):
    """The invariance above is not vacuous: with the masks off the same fixed latents no longer make it invariant."""
    with _switches(stage2, image=False, language=False):
        base = stage2._lit_prefix_pass(_pre(_BASE))
        reference = _velocity("stage2", base)
        for name in ("images", "text"):
            other = stage2._lit_prefix_pass(_pre(_OBS[name]))
            assert _differs(_velocity("stage2", other, name, extra=base.extra), reference) > 0.0, name


@pytest.mark.parametrize(("mask_image", "mask_language"), [(True, False), (False, True)])
def test_each_mask_flag_hides_only_its_own_columns(stage2, mask_image, mask_language):
    """Overwrite cache columns directly (inputs cannot isolate a role: image and language tokens attend each other
    inside the prefix pass): hidden columns change nothing, visible ones do."""
    with _switches(stage2, image=mask_image, language=mask_language):
        prefix = stage2._lit_prefix_pass(_pre(_BASE))
    reference = _velocity("stage2", prefix)
    noise = tuple(jax.random.normal(jax.random.key(i), c.shape, c.dtype) * 3.0 for i, c in enumerate(prefix.kv_cache))

    def overwritten(columns):
        where = columns[None, :, :, None, None]
        return prefix._replace(
            kv_cache=tuple(jnp.where(where, n, c) for n, c in zip(noise, prefix.kv_cache, strict=True))
        )

    hidden = ~prefix.visible
    _equal(_velocity("stage2", overwritten(hidden)), reference)
    assert _differs(_velocity("stage2", overwritten(prefix.visible)), reference) > 0.0
    assert bool(jnp.any(prefix.visible))
    assert bool(jnp.any(hidden))


# ---- the goal never reaches tokens or the state ----


def test_the_goal_never_reaches_the_tokens_the_state_or_the_stage2_action_rows(stage2):
    base, other = _pre(_BASE), _pre(_GOAL_ONLY)
    _equal(other.state, base.state)
    _equal(other.tokenized_prompt, base.tokenized_prompt)
    _equal(other.tokenized_prompt_mask, base.tokenized_prompt_mask)
    for got, want in zip(stage2.embed_prefix(other), stage2.embed_prefix(base), strict=True):
        _equal(got, want)
    x_t, time = _x_t_and_time()
    for got, want in zip(stage2.embed_suffix(other, x_t, time), stage2.embed_suffix(base, x_t, time), strict=True):
        _equal(got, want)
    # in stage 2 the goal is a target only: the action loss cannot depend on it, the pose loss does
    actions = _utils.make_actions(1)
    loss, aux = stage2.compute_loss_and_aux(RNG, _BASE, actions)
    loss_other, aux_other = stage2.compute_loss_and_aux(RNG, _GOAL_ONLY, actions)
    _equal(loss, loss_other)
    assert float(aux["pose_loss"]) != float(aux_other["pose_loss"])
    assert float(aux["pose_copy_baseline"]) != float(aux_other["pose_copy_baseline"])
    # the model's state input is the full normalised state, not the goal and not a slice of it
    assert _pre(_BASE).state.shape == (B, ACTION_DIM)
    _equal(_velocity("stage2", _prefix("stage2", "goal"), "goal"), _velocity("stage2", _prefix("stage2", "base")))


def test_stage1_reads_the_goal_through_kv_columns_only(stage1):
    base, other = _pre(_BASE), _pre(_GOAL_ONLY)
    for got, want in zip(stage1.embed_prefix_text(other), stage1.embed_prefix_text(base), strict=True):
        _equal(got, want)
    prefix, shifted = _prefix("stage1", "base"), _prefix("stage1", "goal")
    _equal(prefix.kv_cache[0], shifted.kv_cache[0])  # the backbone pass never sees the goal
    reference = _velocity("stage1", prefix)
    assert _differs(_velocity("stage1", shifted, "goal"), reference) > 0.0  # but the action rows do, via the columns


# ---- stage 1 structure ----


def test_stage1_has_no_images_and_the_goal_kv_is_the_encoders_and_shared_by_every_layer(stage1):
    nan_images = {k: jnp.full_like(v, jnp.nan) for k, v in _BASE.images.items()}
    prefix = stage1._lit_prefix_pass(_pre(_BASE.replace(images=nan_images)))
    assert prefix.kv_cache[0].shape == (4, B, TEXT, 1, 16), "language tokens only: no image columns"
    assert prefix.latents is None
    clean = _prefix("stage1", "base")
    for got, want in zip(prefix, clean, strict=True):
        if isinstance(want, tuple):
            for a, b in zip(got, want, strict=True):
                _equal(a, b)
        elif want is not None:
            _equal(got, want)
    # NaN images changed nothing, so SigLIP was never consulted
    _equal(_velocity("stage1", prefix), _velocity("stage1", clean))
    goal, valid = stage1._lit_goal(_pre(_BASE))
    assert valid.tolist() == [True, True]
    key, value = stage1.lit_goal_encoder.project_kv(stage1.lit_goal_encoder(goal))
    k, v, visible = clean.extra
    assert k.shape == (4, B, 2, 1, 16)
    for layer in range(4):
        _equal(k[layer], key.reshape(B, 2, 1, 16))
        _equal(v[layer], value.reshape(B, 2, 1, 16))
    assert bool(jnp.all(visible))


def test_stage1_hides_a_padded_goal_from_the_action_rows(stage1):
    padded = _BASE.replace(lit_goal_mask=jnp.asarray([True, False]))
    prefix = stage1._lit_prefix_pass(_pre(padded))
    assert np.asarray(prefix.extra[2]).tolist() == [[True, True], [False, False]]
    x_t, time = _x_t_and_time()

    def velocity(goal):
        obs = _pre(padded.replace(lit_goal=goal))
        p = stage1._lit_prefix_pass(obs)
        return stage1._suffix_velocity(obs, x_t, time, p.kv_cache, p.visible, p.offset, p.extra)

    reference = velocity(_BASE.lit_goal)
    wild = _BASE.lit_goal.at[1].set(jnp.nan)
    moved = velocity(wild)
    _equal(moved[1], reference[1])  # the padded sample ignores its goal, even a NaN one
    assert bool(jnp.all(jnp.isfinite(moved)))
    assert _differs(velocity(_BASE.lit_goal.at[0].add(1.0))[0], reference[0]) > 0.0  # the valid one does not


def test_stage1_needs_a_goal_and_stage2_does_not_for_the_plain_loss(stage1, stage2):
    no_goal = _BASE.replace(lit_goal=None, lit_goal_mask=None)
    actions = _utils.make_actions(1)
    with pytest.raises(ValueError, match="lit_goal"):
        stage1.compute_loss(RNG, no_goal, actions)
    loss = stage2.compute_loss(RNG, no_goal, actions)
    assert loss.shape == (B, 4)
    with pytest.raises(ValueError, match="lit_goal"):
        stage2.compute_loss_and_aux(RNG, no_goal, actions)


def test_a_goal_without_its_mask_is_refused(stage1, stage2):
    """A zero-padded goal with a forgotten mask would otherwise pass as a real target."""
    unmasked = _BASE.replace(lit_goal_mask=None)
    actions = _utils.make_actions(1)
    for model in (stage1, stage2):
        with pytest.raises(ValueError, match="lit_goal_mask"):
            model._lit_goal(_pre(unmasked))
    with pytest.raises(ValueError, match="lit_goal_mask"):
        stage1.compute_loss(RNG, unmasked, actions)
    with pytest.raises(ValueError, match="lit_goal_mask"):
        stage2.compute_loss_and_aux(RNG, unmasked, actions)


# ---- losses ----


def test_compute_loss_returns_the_action_loss_with_the_stock_shape(stage1, stage2):
    actions = _utils.make_actions(1)
    for model in (stage1, stage2):
        loss = model.compute_loss(RNG, _BASE, actions)
        both, aux = model.compute_loss_and_aux(RNG, _BASE, actions)
        assert loss.shape == (B, _utils.ACTION_HORIZON)
        assert loss.dtype == jnp.float32
        _equal(loss, both)
        assert bool(jnp.all(jnp.isfinite(loss)))
        assert all(v.shape == () for v in aux.values())
    assert sorted(aux) == ["pose_copy_baseline", "pose_loss"]
    assert stage2.lit_pose_weight == 0.3  # the weight lives on the model for the train step; the losses are unweighted
    assert sorted(stage1.compute_loss_and_aux(RNG, _BASE, actions)[1]) == ["pose_copy_baseline"]


def test_pose_losses_match_a_numpy_recomputation_over_the_valid_goals(stage2):
    actions = _utils.make_actions(1)
    obs = _obs(0, goal_seed=5, goal_mask=(True, False))
    _, aux = stage2.compute_loss_and_aux(RNG, obs, actions)
    pre = _pre(obs)
    prefix = stage2._lit_prefix_pass(pre)
    predicted = np.asarray(stage2.lit_pose_decoder(prefix.latents), np.float64)
    goal = np.asarray(obs.lit_goal, np.float64)[:, list(GOAL_DIMS)]
    state = np.asarray(pre.state, np.float64)[:, list(GOAL_DIMS)]
    per_sample = ((predicted - goal) ** 2).mean(-1)
    copy_per_sample = ((state - goal) ** 2).mean(-1)
    np.testing.assert_allclose(float(aux["pose_loss"]), per_sample[0], rtol=1e-5)
    np.testing.assert_allclose(float(aux["pose_copy_baseline"]), copy_per_sample[0], rtol=1e-5)
    both = _obs(0, goal_seed=5)
    _, aux_both = stage2.compute_loss_and_aux(RNG, both, actions)
    np.testing.assert_allclose(float(aux_both["pose_loss"]), per_sample.mean(), rtol=1e-5)


def test_padded_goals_are_ignored_by_the_pose_loss_and_its_gradient(stage2):
    actions = _utils.make_actions(1)

    def pose_loss_and_grads(goal):
        obs = _obs(0, goal_mask=(True, False)).replace(lit_goal=goal)
        return nnx.value_and_grad(
            lambda m: m.compute_loss_and_aux(RNG, obs, actions)[1]["pose_loss"], argnums=nnx.DiffState(0, nnx.Param)
        )(stage2)

    goal = _goal(0)
    base_loss, base_grads = pose_loss_and_grads(goal)
    for wild in (goal.at[1].set(1e6), goal.at[1].set(jnp.nan)):
        loss, grads = pose_loss_and_grads(wild)
        _equal(loss, base_loss)
        for path, var in base_grads.flat_state().items():
            _equal(grads.flat_state()[path].value, var.value)
    # and when every goal is padding the loss is 0, not NaN
    nothing = _obs(0, goal_mask=(False, False)).replace(lit_goal=jnp.full((B, ACTION_DIM), jnp.nan))
    _, aux = stage2.compute_loss_and_aux(RNG, nothing, actions)
    assert float(aux["pose_loss"]) == 0.0
    assert float(aux["pose_copy_baseline"]) == 0.0


def test_loss_is_finite_in_bfloat16_and_the_extra_columns_follow_the_cache_dtype():
    config = _utils.make_tiny_config(lit="stage2", dtype="bfloat16")
    model = _utils.randomize_zero_init(_utils.build_model(config))
    obs = _obs(0, config=config)
    loss, aux = model.compute_loss_and_aux(RNG, obs, _utils.make_actions(1, config=config))
    assert bool(jnp.all(jnp.isfinite(loss)))
    assert all(bool(jnp.isfinite(v)) for v in aux.values())
    prefix = model._lit_prefix_pass(_pre(obs))
    assert prefix.kv_cache[0].dtype == jnp.bfloat16
    assert prefix.extra[0].dtype == jnp.bfloat16
    assert prefix.latents.dtype == jnp.bfloat16


def test_jit_loss_matches_eager(stage2):
    actions = _utils.make_actions(1)
    eager = stage2.compute_loss(RNG, _BASE, actions)
    jitted = _utils.jit_loss(stage2)(RNG, _BASE, actions)
    print(f"jit vs eager loss: {_differs(jitted, eager) / float(jnp.max(jnp.abs(eager))):.2e} relative")
    assert _differs(jitted, eager) <= _REL * float(jnp.max(jnp.abs(eager)))


# ---- gradients ----


def _grad_maxes(fn, model, wrt=nnx.Param) -> dict:
    grads = nnx.grad(fn, argnums=nnx.DiffState(0, wrt))(model)
    return {path: np.asarray(var.value) for path, var in ((_path(p), v) for p, v in grads.flat_state().items())}


def _nonzero(array) -> bool:
    return bool(np.any(array != 0))


def _is_expert(path: str) -> bool:
    return "llm" in path and "_1" in path


def _is_backbone_layer(path: str) -> bool:
    return path.startswith("PaliGemma/llm/layers/") and "_1" not in path


_INERT = "/attn/k/bias"  # softmax over the keys is invariant to a per-query constant, so this gradient is zero


def test_stage2_action_loss_trains_backbone_expert_and_every_aggregator_group(stage2):
    actions = _utils.make_actions(1)
    grads = _grad_maxes(lambda m: jnp.mean(m.compute_loss(RNG, _BASE, actions)), stage2)
    depth = 4
    for path, g in grads.items():
        assert bool(np.all(np.isfinite(g))), path
        if _is_backbone_layer(path):
            # layer i's input feeds the aggregator for i = 0..depth-1, and depends on layers 0..i-1 only
            assert g.shape[0] == depth
            assert _nonzero(g[: depth - 1]), path
            assert not _nonzero(g[depth - 1]), path
        elif _is_expert(path) or path.startswith(("action_", "time_mlp")):
            assert _nonzero(g), path
        elif path.startswith("lit_aggregator"):
            if path.endswith(_INERT):
                continue
            for group in range(2):
                assert _nonzero(g[group] if g.ndim > 1 and g.shape[0] == 2 else g), (path, group)
        elif path.startswith("lit_pose_decoder"):
            assert not _nonzero(g), path  # the pose head is not on the action path
    assert _nonzero(grads["PaliGemma/llm/embedder/input_embedding"])
    assert not _nonzero(grads["PaliGemma/llm/final_norm/scale"])
    image_grads = {p: g for p, g in grads.items() if p.startswith("PaliGemma/img/")}
    assert _nonzero(image_grads["PaliGemma/img/head/kernel"])
    assert sum(_nonzero(g) for g in image_grads.values()) > 0.5 * len(image_grads)
    assert _nonzero(grads["lit_aggregator/queries"])


def test_the_aggregator_key_bias_is_inert(stage2):
    actions = _utils.make_actions(1)
    base = stage2.compute_loss(RNG, _BASE, actions)
    graphdef, state = nnx.split(stage2)
    flat = state.flat_state()
    shifted = {
        path: (
            var.replace(value=var.value + 1.0) if _path(path).endswith(_INERT) and _is_lit_path(_path(path)) else var
        )
        for path, var in flat.items()
    }
    assert sum(1 for p in flat if _path(p).endswith(_INERT) and _is_lit_path(_path(p))) == 3
    moved = nnx.merge(graphdef, nnx.State.from_flat_path(shifted)).compute_loss(RNG, _BASE, actions)
    print(f"key bias +1 changes the loss by {_differs(moved, base):.2e} (scale {float(jnp.max(jnp.abs(base))):.2e})")
    assert _differs(moved, base) <= _REL * float(jnp.max(jnp.abs(base)))


def test_stage2_pose_loss_trains_aggregator_and_decoder_not_the_expert(stage2):
    actions = _utils.make_actions(1)
    grads = _grad_maxes(lambda m: m.compute_loss_and_aux(RNG, _BASE, actions)[1]["pose_loss"], stage2)
    for path, g in grads.items():
        assert bool(np.all(np.isfinite(g))), path
        if path.startswith("lit_pose_decoder"):
            assert _nonzero(g), path
        elif path.startswith("lit_aggregator/groups/to_"):
            assert not _nonzero(g), path  # the latent K/V projections are not on the pose path
        elif path.startswith("lit_aggregator"):
            if not path.endswith(_INERT):
                assert _nonzero(g), path
        elif _is_expert(path) or path.startswith(("action_", "time_mlp")):
            assert not _nonzero(g), path
        elif _is_backbone_layer(path):
            assert _nonzero(g[:3]), path
    assert _nonzero(grads["PaliGemma/llm/embedder/input_embedding"])
    assert _nonzero(grads["PaliGemma/img/head/kernel"])


def test_stage1_trains_the_goal_encoder_and_the_expert_only(stage1):
    config = _utils.make_tiny_config(lit="stage1")
    trainable = nnx.All(nnx.Param, nnx.Not(config.get_freeze_filter()))
    actions = _utils.make_actions(1)
    grads = _grad_maxes(lambda m: jnp.mean(m.compute_loss(RNG, _BASE, actions)), stage1, trainable)
    assert grads
    groups = {"goal_encoder": 0, "expert": 0, "projections": 0}
    for path, g in grads.items():
        assert _nonzero(g), path
        if path.startswith("lit_goal_encoder/"):
            groups["goal_encoder"] += 1
        elif _is_expert(path):
            groups["expert"] += 1
        else:
            assert path.startswith(("action_in_proj", "action_out_proj", "time_mlp_")), path
            groups["projections"] += 1
    assert all(groups.values()), groups
    all_paths = set(_flat(nnx.state(stage1, nnx.Param)))
    for path in all_paths - set(grads):
        assert not _is_expert(path), path
        assert not path.startswith(("lit_", "action_", "time_mlp")), path
    # without the filter the backbone would receive gradient: the filter is what keeps it frozen
    everything = _grad_maxes(lambda m: jnp.mean(m.compute_loss(RNG, _BASE, actions)), stage1)
    assert _nonzero(everything["PaliGemma/llm/embedder/input_embedding"])
    assert "PaliGemma/llm/embedder/input_embedding" not in grads
    assert not any(p.startswith("lit_aggregator") for p in all_paths)


# ---- the stage 1 -> stage 2 hand-off and the freeze filter ----


def test_stage1_to_stage2_key_difference_is_exactly_the_goal_encoder():
    stage1_manifest, stage2_manifest = _stage_model("stage1")[0], _stage_model("stage2")[0]
    only1, only2 = set(stage1_manifest) - set(stage2_manifest), set(stage2_manifest) - set(stage1_manifest)
    assert only1
    assert all(p.startswith("lit_goal_encoder/") for p in only1)
    assert {p.split("/")[0] for p in only2} == {"lit_aggregator", "lit_pose_decoder"}
    shared = set(stage1_manifest) & set(stage2_manifest)
    assert not any(_is_lit_path(p) for p in shared)
    for path in shared:
        assert stage1_manifest[path] == stage2_manifest[path], path
    print(f"stage1 only: {len(only1)} leaves; stage2 only: {len(only2)} leaves; shared: {len(shared)}")


def test_loading_stage1_params_into_stage2_drops_the_goal_encoder_and_keeps_fresh_lit_params():
    stage1_params = _stage_model("stage1")[1]
    stage2_params = _stage_model("stage2")[1]
    loaded = nnx.state(stage1_params, nnx.Param).to_pure_dict()
    fresh = nnx.state(stage2_params, nnx.Param).to_pure_dict()
    merged = weight_loaders._merge_params(loaded, fresh, missing_regex=".*lit_.*")
    flat = _flatten(merged)
    flat_fresh = _flatten(fresh)
    assert sorted(flat) == sorted(flat_fresh), "exactly the stage-2 tree: the goal encoder is gone, nothing is missing"
    for path, value in flat.items():
        source = flat_fresh[path] if _is_lit_path(path) else _flatten(loaded)[path]
        _equal(value, source)
    # the checkpoint loader as it is today only fills lora keys, so the new lit_* leaves would be missing
    stock = weight_loaders._merge_params(loaded, fresh, missing_regex=".*lora.*")
    assert {p for p in flat_fresh if _is_lit_path(p)} == set(flat_fresh) - set(_flatten(stock))


def _flatten(tree) -> dict:
    return flax.traverse_util.flatten_dict(tree, sep="/")


def _real_config(stage="stage2", *, lora=False, **overrides):
    variants = {"paligemma_variant": "gemma_2b_lora", "action_expert_variant": "gemma_300m_lora"} if lora else {}
    return pi0_config.Pi0Config(
        pi05=True,
        action_horizon=30,
        max_token_len=200,
        lit=stage,
        lit_goal_dims=tuple(range(7)),
        **variants,
        **overrides,
    )


@functools.cache
def _real_leaves(stage: str, *, lora: bool) -> dict:
    """path -> shape of every Param at real dimensions, from eval_shape only (nothing is allocated)."""
    model = nnx.eval_shape(lambda: _real_config(stage, lora=lora).create(jax.random.key(0)))
    return {p: v.shape for p, v in ((_path(p), v.value) for p, v in nnx.state(model, nnx.Param).flat_state().items())}


def _count(shapes: dict) -> int:
    return int(sum(int(np.prod(s)) for s in shapes.values()))


def _expected_aggregator_params(dim=768, context=2048, inner=3072, groups=6, latents=100, kv=256) -> int:
    norm = 2 * dim
    attn = lambda kv_in: 2 * (dim * dim + dim) + 2 * (kv_in * dim + dim)  # noqa: E731
    ffn = dim * inner + inner + inner * dim + dim
    self_block = norm + attn(dim) + norm + ffn
    cross_block = norm + 2 * context + attn(context) + norm + ffn
    return latents * dim + groups * (self_block + 2 * cross_block + 2 * dim * kv)


def test_real_dimension_lit_parameters(capsys):
    stage2, stage1, control = (
        _real_leaves("stage2", lora=False),
        _real_leaves("stage1", lora=False),
        _real_leaves("off", lora=False),
    )
    lit2 = {p: s for p, s in stage2.items() if _is_lit_path(p)}
    lit1 = {p: s for p, s in stage1.items() if _is_lit_path(p)}
    assert {p: s for p, s in stage2.items() if not _is_lit_path(p)} == control
    assert {p: s for p, s in stage1.items() if not _is_lit_path(p)} == control
    aggregator = {p: s for p, s in lit2.items() if p.startswith("lit_aggregator")}
    assert _count(aggregator) == _expected_aggregator_params()
    assert lit2["lit_aggregator/queries"] == (100, 768)
    assert lit2["lit_aggregator/groups/to_key/kernel"] == (6, 768, 256)
    assert lit2["lit_aggregator/groups/semantic_block/context_norm/scale"] == (6, 2048)
    assert lit2["lit_pose_decoder/fc1/kernel"] == (8 * 768, 512)
    assert lit1["lit_goal_encoder/fc3/kernel"] == (512, 8 * 768)
    assert lit1["lit_goal_encoder/to_key/kernel"] == (768, 256)
    with capsys.disabled():
        print(
            f"\nreal dims: backbone+expert params {_count(control):,}; stage2 lit_* {_count(lit2):,} "
            f"(aggregator {_count(aggregator):,}, pose decoder {_count(lit2) - _count(aggregator):,}); "
            f"stage1 lit_* {_count(lit1):,}"
        )


@functools.cache
def _real_traced(stage: str):
    """The LIT prefix pass, compute_loss_and_aux and (stage 2) sample_actions traced at real dimensions: ShapeDtypeStructs
    only, nothing is allocated."""
    config = _real_config(stage)

    def trace():
        model = config.create(jax.random.key(0))
        observation, actions = config.fake_obs(1), config.fake_act(1)
        out = {
            "prefix": model._lit_prefix_pass(_pre(observation)),
            "loss": model.compute_loss_and_aux(jax.random.key(1), observation, actions),
        }
        if stage == "stage2":
            out["actions"] = model.sample_actions(jax.random.key(2), observation, num_steps=2)
        return out

    return nnx.eval_shape(trace)


def test_real_dimension_stage2_loss_and_sampling_trace_with_the_expected_shapes():
    traced = _real_traced("stage2")
    prefix, (loss, aux) = traced["prefix"], traced["loss"]
    layers, s = 18, CAMERAS * IMAGE_TOKENS + 200
    assert [x.shape for x in prefix.kv_cache] == [(layers, 1, s, 1, 256)] * 2
    assert prefix.kv_cache[0].dtype == jnp.bfloat16
    # the stacked per-layer prefix inputs the aggregator reads are (18, 1, 968, 2048); what comes out is K=100 latents
    assert [x.shape for x in prefix.extra[:2]] == [(layers, 1, 100, 1, 256)] * 2
    assert prefix.extra[0].dtype == prefix.kv_cache[0].dtype
    assert prefix.extra[2].shape == (1, 100)
    assert prefix.latents.shape == (1, 100, 768)
    assert (prefix.visible.shape, prefix.count.shape, prefix.offset.shape) == ((1, s), (1,), (1,))
    assert loss.shape == (1, 30)
    assert sorted(aux) == ["pose_copy_baseline", "pose_loss"]
    assert all(v.shape == () for v in aux.values())
    assert traced["actions"].shape == (1, 30, 32)


def test_real_dimension_stage1_loss_traces_with_the_expected_shapes():
    traced = _real_traced("stage1")
    prefix, (loss, aux) = traced["prefix"], traced["loss"]
    assert [x.shape for x in prefix.kv_cache] == [(18, 1, 200, 1, 256)] * 2, "the language only: no image columns"
    assert [x.shape for x in prefix.extra[:2]] == [(18, 1, 8, 1, 256)] * 2, "8 goal tokens"
    assert prefix.latents is None
    assert loss.shape == (1, 30)
    assert sorted(aux) == ["pose_copy_baseline"]


def _trainable(config) -> tuple[dict, dict]:
    model = nnx.eval_shape(lambda: config.create(jax.random.key(0)))
    state = nnx.state(model, nnx.Param)
    flat = state.flat_state()
    train = set(state.filter(nnx.All(nnx.Param, nnx.Not(config.get_freeze_filter()))).flat_state())
    return (
        {_path(p): flat[p].value.shape for p in flat if p in train},
        {_path(p): flat[p].value.shape for p in flat if p not in train},
    )


@pytest.mark.parametrize(
    ("name", "stage", "lora"),
    [("litctl", "off", False), ("lit1", "stage1", False), ("lit2", "stage2", False), ("litlite", "stage2", True)],
)
def test_freeze_filter_at_real_dimensions(name, stage, lora, capsys):
    config = _real_config(stage, lora=lora)
    train, frozen = _trainable(config)
    lit_train = {p for p in train if _is_lit_path(p)}
    with capsys.disabled():
        print(
            f"\nfreeze filter {name} ({stage}{', lora' if lora else ''}): trainable {len(train)} leaves / "
            f"{_count(train):,} params, frozen {len(frozen)} leaves / {_count(frozen):,} params, "
            f"of which lit_* trainable {len(lit_train)} leaves / {_count({p: train[p] for p in lit_train}):,}"
        )
    assert set(train) | set(frozen) == set(_real_leaves(stage, lora=lora))
    assert lit_train == {p for p in _real_leaves(stage, lora=lora) if _is_lit_path(p)}, "every lit_* leaf trains"
    if name in ("litctl", "lit2"):
        assert not frozen
    if name == "lit1":
        assert all(
            p.startswith(("lit_goal_encoder/", "action_in_proj", "action_out_proj", "time_mlp_"))
            or (_is_expert(p) and p.startswith("PaliGemma/llm/"))
            for p in train
        )
        assert all(p.startswith(("PaliGemma/img/", "PaliGemma/llm/")) and not _is_expert(p) for p in frozen)
        assert any(p.startswith("PaliGemma/img/") for p in frozen)
    if name == "litlite":
        assert frozen
        assert all("lora" not in p for p in frozen)
        assert all(p.startswith("PaliGemma/llm/") for p in frozen)
        assert any("lora" in p for p in train)


# ---- the valid-token count must not reach the action rows through their RoPE positions ----
#
# The latent K/V carry no RoPE while the action rows' queries are rotated by their absolute position, so an action
# row's logits against a latent depend on the position it starts from. Started from the number of valid prefix tokens,
# that count (prompt length, camera masks) reaches rows that are otherwise blind to the whole prefix. The probes need
# latent K/V that are held constant AND non-zero: against a zero key every logit is 0 whatever the rotation, and
# perturbing observation.state is vacuous under pi05 (the state is in the prompt).


@contextlib.contextmanager
def _constant_latents(model):
    """The aggregator returns the base observation's latents and latent K/V whatever it is fed."""
    base = _prefix("stage2", "base")
    keys, values = (x.reshape(*x.shape[:3], -1) for x in base.extra[:2])
    assert float(jnp.max(jnp.abs(keys))) > 0.0
    assert float(jnp.max(jnp.abs(values))) > 0.0
    real = model.lit_aggregator
    model.lit_aggregator = lambda *_: (base.latents, keys, values)
    try:
        yield
    finally:
        model.lit_aggregator = real


def _with_prompt_lengths(observation, lengths):
    mask = np.arange(TEXT)[None, :] < np.asarray(lengths)[:, None]
    tokens = np.where(mask, np.asarray(observation.tokenized_prompt), 0).astype(np.int32)
    return observation.replace(tokenized_prompt=jnp.asarray(tokens), tokenized_prompt_mask=jnp.asarray(mask))


def _with_camera_off(observation, camera, sample=None):
    mask = np.asarray(observation.image_masks[camera]).copy()
    mask[slice(None) if sample is None else sample] = False
    return observation.replace(image_masks={**observation.image_masks, camera: jnp.asarray(mask)})


def _count_variants(base):
    """Observations that change the valid-token count of the prefix and nothing else a hidden column could carry into
    the action rows (the hidden columns' contents move too, but they are masked: that is what is under test)."""
    return {
        "prompt 8 -> 7 valid tokens (sample 0)": _with_prompt_lengths(base, (TEXT - 1, TEXT - 3)),
        "prompt 8 -> 4 and 5 -> 8 valid tokens": _with_prompt_lengths(base, (TEXT - 4, TEXT)),
        "prompt 2 valid tokens": _with_prompt_lengths(base, (2, 2)),
        "camera 0 off (sample 0)": _with_camera_off(base, "base_0_rgb", 0),
        "camera 1 off (both samples)": _with_camera_off(base, "left_wrist_0_rgb"),
        "all three cameras off (sample 0)": _with_camera_off(
            _with_camera_off(_with_camera_off(base, "base_0_rgb", 0), "left_wrist_0_rgb", 0), "right_wrist_0_rgb", 0
        ),
    }


def test_hard_mask_velocity_is_exactly_independent_of_the_valid_token_count():
    model = _stage_model("stage2")[1]
    x_t, time = _x_t_and_time()

    def velocity(prefix, observation, offset):
        args = (prefix.kv_cache, prefix.visible, offset, prefix.extra)
        return model._suffix_velocity(_pre(observation), x_t, time, *args)

    with _constant_latents(model):
        base = model._lit_prefix_pass(_pre(_BASE))
        reference = velocity(base, _BASE, base.offset)
        raw_reference = velocity(base, _BASE, base.count)
        for name, observation in _count_variants(_BASE).items():
            prefix = model._lit_prefix_pass(_pre(observation))
            assert not np.array_equal(np.asarray(prefix.count), np.asarray(base.count)), name
            assert not bool(jnp.any(prefix.visible)), name
            moved = _differs(velocity(prefix, observation, prefix.offset), reference)
            # the probe has power: started from the raw count, the same constant latents follow the count
            leaked = _differs(velocity(prefix, observation, prefix.count), raw_reference)
            print(f"{name}: velocity moved by {moved:.1e}; from the raw count it would move by {leaked:.1e}")
            assert moved == 0.0, f"{name}: the velocity moved by {moved:.3e}"
            assert leaked > 0.0, name
            _equal(prefix.offset, jnp.zeros_like(prefix.count))


def test_hard_mask_loss_is_exactly_independent_of_the_valid_token_count(monkeypatch):
    model = _stage_model("stage2")[1]
    actions = _utils.make_actions(1)
    rng = jax.random.key(1)
    variants = list(_count_variants(_BASE).values())
    with _constant_latents(model):
        reference = model.compute_loss(rng, _BASE, actions)
        for observation in variants:
            _equal(model.compute_loss(rng, observation, actions), reference)
        # the probe has power: with the raw count as the start the loss follows the count
        monkeypatch.setattr(_pi0.Pi0, "_lit_suffix_offset", lambda self, count: count)
        leaked = _differs(model.compute_loss(rng, variants[0], actions), model.compute_loss(rng, _BASE, actions))
    print(f"loss from the raw count would move by {leaked:.1e}")
    assert leaked > 0.0


@pytest.mark.parametrize(("mask_image", "mask_language"), [(True, False), (False, True), (False, False)])
def test_with_a_prefix_column_visible_the_suffix_starts_from_the_valid_token_count(stage2, mask_image, mask_language):
    """Only the both-masks-on setting hides the whole prefix, so only there is the start constant: not silently so
    for the one-sided ablations or the unmasked model, whose visible columns are rotated at their own positions."""
    shorter = _with_prompt_lengths(_BASE, (TEXT - 1, TEXT - 3))
    with _switches(stage2, image=mask_image, language=mask_language):
        prefix = stage2._lit_prefix_pass(_pre(_BASE))
        shorter_prefix = stage2._lit_prefix_pass(_pre(shorter))
    assert bool(jnp.any(prefix.visible))
    _equal(prefix.offset, prefix.count)
    _equal(shorter_prefix.offset, shorter_prefix.count)
    counts = [PREFIX_LEN, PREFIX_LEN - IMAGE_TOKENS - (TEXT - TEXT_LENGTHS[1])]
    assert np.asarray(prefix.offset).tolist() == counts
    # one prompt token fewer for sample 0, none fewer for sample 1
    assert np.asarray(shorter_prefix.offset).tolist() == [counts[0] - 1, counts[1]]


def test_stage1_starts_the_action_rows_from_the_valid_token_count():
    prefix = _prefix("stage1", "base")
    _equal(prefix.offset, prefix.count)
    assert np.asarray(prefix.offset).tolist() == list(TEXT_LENGTHS)


# ---- the sampling path: sample_actions ----
#
# Sampling observations are the plain ones lit_golden uses (no goal: sampling never reads it), so a lit="off" model and
# a stage-2 model with the same backbone parameters see exactly the same prefix and can be compared with the stock
# sampler. openpi has no memory dict and returns the actions only. Each prefix pass runs SigLIP eagerly, hence the caches.

_STEPS = _gen.NUM_STEPS
_DT = -1.0 / _STEPS
_SAMPLE_RNG = jax.random.key(_gen.SAMPLE_RNG_SEED)
_LATENTS = _utils.TINY_LIT_DEFAULTS["lit_num_latents"]


def _sample_obs(call: int = 0, **kwargs):
    return _utils.make_observation(_gen.OBSERVATION_SEEDS[call], **kwargs)


def _sample_noise(call: int = 0):
    return _utils.fixed_noise(_gen.NOISE_SEEDS[call])


def _sample(model, observation, noise, *, steps=_STEPS):
    return model.sample_actions(_SAMPLE_RNG, observation, num_steps=steps, noise=noise)


@functools.cache
def _sampled(call: int):
    return np.asarray(_sample(_stage_model("stage2")[1], _sample_obs(call), _sample_noise(call)))


@functools.cache
def _sampling_prefix(call: int):
    """The (extended cache, visible, offset) the sampler builds, from the training path's own pieces."""
    model = _stage_model("stage2")[1]
    prefix = model._lit_prefix_pass(_pre(_sample_obs(call)))
    cache, visible = model._lit_extend_cache(prefix.kv_cache, prefix.visible, prefix.extra)
    return cache, visible, prefix.offset


def _denoise(model, call, cache, visible, offset, observation=None):
    observation = _pre(_sample_obs(call) if observation is None else observation)
    return model._denoise(observation, _sample_noise(call), _DT, cache, visible, offset)


def _with_latents_of(cache, other):
    """`cache` with its latent columns (the last _LATENTS) replaced by those of `other`."""
    return tuple(
        jnp.concatenate([c[:, :, :-_LATENTS], o[:, :, -_LATENTS:]], axis=2) for c, o in zip(cache, other, strict=True)
    )


@exact_equality
def test_lit_off_sampling_equals_the_baseline_fixture_and_never_enters_the_lit_path(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the LIT sampling path ran for lit='off'")

    monkeypatch.setattr(_pi0.Pi0, "_sample_actions_lit", refuse)
    actions = _gen.run_sample("f32_rand_eager")
    with np.load(_gen.FIXTURE_DIR / "baseline_actions.npz") as fixture:
        expected = {k: v for k, v in fixture.items() if k.startswith("f32_rand_eager__")}
    assert sorted(actions) == sorted(expected) == [_gen.actions_key("f32_rand_eager", i) for i in (1, 2)]
    for key, want in expected.items():
        assert actions[key].dtype == want.dtype
        _equal(actions[key], want)


def test_stage1_cannot_sample_and_lit_off_has_no_lit_sampler(stage1):
    observation = _sample_obs(0)
    with pytest.raises(ValueError, match="training-only"):
        stage1.sample_actions(_SAMPLE_RNG, observation)
    with pytest.raises(ValueError, match="stage2"):
        _gen.get_model("float32", "rand")._sample_actions_lit(_SAMPLE_RNG, observation, num_steps=2, noise=None)


def test_sample_actions_returns_the_actions_only_and_is_the_shared_denoise_over_the_lit_prefix():
    model = _stage_model("stage2")[1]
    for call in range(2):
        actions = _sampled(call)
        assert actions.shape == (B, _utils.ACTION_HORIZON, ACTION_DIM)
        assert actions.dtype == np.float32
        assert np.isfinite(actions).all()
        cache, visible, offset = _sampling_prefix(call)
        _equal(_denoise(model, call, cache, visible, offset), actions)


def test_the_prefix_pass_and_the_aggregator_run_once_per_sampling_call(monkeypatch):
    model = _stage_model("stage2")[1]
    calls = {"prefix": 0, "aggregator": 0}
    real_prefix, real_aggregator = _pi0.Pi0._lit_prefix_pass, model.lit_aggregator

    def counting_prefix(self, observation):
        calls["prefix"] += 1
        return real_prefix(self, observation)

    def counting_aggregator(*args):
        calls["aggregator"] += 1
        return real_aggregator(*args)

    monkeypatch.setattr(_pi0.Pi0, "_lit_prefix_pass", counting_prefix)
    model.lit_aggregator = counting_aggregator
    try:
        _sample(model, _sample_obs(0), _sample_noise(0), steps=5)
    finally:
        model.lit_aggregator = real_aggregator
    assert calls == {"prefix": 1, "aggregator": 1}, "five denoising steps, one prefix pass and one set of latents"


def test_sampling_noise_comes_from_rng_when_none_is_given():
    model = _stage_model("stage2")[1]
    observation = _sample_obs(0)
    drawn = jax.random.normal(_SAMPLE_RNG, (B, _utils.ACTION_HORIZON, ACTION_DIM))
    _equal(model.sample_actions(_SAMPLE_RNG, observation, num_steps=_STEPS), _sample(model, observation, drawn))


@pytest.mark.parametrize("steps", [1, 2, 7])
def test_any_number_of_denoising_steps_samples_finite_actions(steps):
    out = _sample(_stage_model("stage2")[1], _sample_obs(0), _sample_noise(0), steps=steps)
    assert out.shape == (B, _utils.ACTION_HORIZON, ACTION_DIM)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_the_masks_are_what_separates_stage2_from_the_stock_sampler(stage2):
    """Both sampled from the same weights: the stock joint prefix reads everything, stage 2 only its latents."""
    observation, noise = _sample_obs(0), _sample_noise(0)
    lit = _sample(stage2, observation, noise)
    stage2.lit = "off"
    try:
        stock = _sample(stage2, observation, noise)
    finally:
        stage2.lit = "stage2"
    assert _differs(lit, stock) > 1e-3


def test_two_pass_with_no_lit_mask_and_no_latents_equals_the_stock_sampler(stage2, capsys):
    """Masks off and the latent columns dropped, the two-pass sampler is the stock joint one, on the same weights."""
    worst = 0.0
    with _switches(stage2, image=False, language=False):
        for call in range(2):
            observation, noise = _sample_obs(call), _sample_noise(call)
            pre = _pre(observation)
            prefix = stage2._lit_prefix_pass(pre)
            two_pass = stage2._denoise(pre, noise, _DT, prefix.kv_cache, prefix.visible, prefix.offset)
            stage2.lit = "off"
            try:
                stock = _sample(stage2, observation, noise)
            finally:
                stage2.lit = "stage2"
            worst = max(worst, _differs(two_pass, stock) / float(np.max(np.abs(stock))))
    with capsys.disabled():
        print(f"\nsampling two-pass vs stock sampler: worst relative difference {worst:.3e} (tolerance {_REL})")
    assert worst <= _REL


def _train_velocity(model, observation, x_t, time):
    prefix = model._lit_prefix_pass(_pre(observation))
    return model._suffix_velocity(
        _pre(observation), x_t, time, prefix.kv_cache, prefix.visible, prefix.offset, prefix.extra
    )


def _serve_velocity_eager(model, observation, x_t, time):
    """The velocity over the cache the sampler builds (latent columns already inside), eagerly."""
    pre = _pre(observation)
    prefix = model._lit_prefix_pass(pre)
    cache, visible = model._lit_extend_cache(prefix.kv_cache, prefix.visible, prefix.extra)
    return model._suffix_velocity(pre, x_t, time, cache, visible, prefix.offset)


def _serve_velocity_sampled(model, observation, noise):
    """The velocity inside sample_actions, recovered from one Euler step from t=1: x_0 = noise - v(noise, 1)."""
    return noise - model.sample_actions(_SAMPLE_RNG, observation, num_steps=1, noise=noise)


# Measured on this CPU with the real SigLIP (the printed lines): eager train vs eager serve velocity is exactly 0 in
# both dtypes (they are the same routine over the same cache); the compiled sampling loop differs from the eager
# training path by float rounding, and in bfloat16 by bf16 rounding (XLA fuses the loop and keeps excess precision).
# The tolerances are those measurements with headroom: float32 1e-5 over 1.2e-7, bfloat16 2e-2 over 3.0e-3 (one bf16
# rounding step is 2**-8 = 3.9e-3 of the value). They are margins around numbers measured on ONE toolchain (arm64 CPU,
# jax 0.5.3, flax 0.10.2, XLA CPU's fusion choices): another backend fuses and rounds differently (an accelerator keeps
# other excess precision), so on one the bfloat16 margin is a hypothesis to re-measure, not a bound the code guarantees.
_VELOCITY_REL = {"float32": 1e-5, "bfloat16": 2e-2}


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_train_and_serve_compute_the_same_velocity(dtype, capsys):
    """The velocity of the training path (_lit_prefix_pass + its own latent columns) and of the sampling path (the
    cache with the latents inside, then the compiled denoising loop), for the same x_t, t and observation."""
    config = _utils.make_tiny_config(lit="stage2", dtype=dtype)
    model = _stage_model("stage2")[1] if dtype == "float32" else _stage_model("stage2", dtype="bfloat16")[1]
    observation = _sample_obs(0, config=config)  # a camera of sample 1 missing, prompts of different lengths
    noise = _utils.fixed_noise(0, config=config)
    time = jnp.ones((B,), jnp.float32)  # the sampler's first step: t = 1, x_t = noise
    train = _train_velocity(model, observation, noise, time)
    serve_eager = _serve_velocity_eager(model, observation, noise, time)
    serve = _serve_velocity_sampled(model, observation, noise)
    scale = float(jnp.max(jnp.abs(train)))
    eager_gap, sampled_gap = _differs(serve_eager, train) / scale, _differs(serve, train) / scale
    with capsys.disabled():
        print(
            f"\ntrain vs serve velocity ({dtype}): eager {eager_gap:.2e}, through sample_actions {sampled_gap:.2e} "
            f"relative to {scale:.3e}"
        )
    assert eager_gap == 0.0
    assert sampled_gap <= _VELOCITY_REL[dtype]
    assert float(jnp.max(jnp.abs(train))) > 0.0


def test_train_and_serve_velocities_agree_under_every_valid_token_count_variant():
    """The offset has to come from the same place in the training pass and the sampling pass (it is one helper): with
    masked cameras and different prompt lengths per sample, the two velocities still agree exactly."""
    model = _stage_model("stage2")[1]
    x_t, time = _x_t_and_time()
    for name, observation in {"plain": _BASE, **_count_variants(_BASE)}.items():
        _equal(_serve_velocity_eager(model, observation, x_t, time), _train_velocity(model, observation, x_t, time))
        print(f"train == serve velocity for {name}")


def test_the_training_loss_is_the_sampling_velocity_against_the_target():
    """End to end: compute_loss, rebuilt from the draws it makes and the SAMPLING path's velocity."""
    model = _stage_model("stage2")[1]
    rng = jax.random.key(0)
    actions = _utils.make_actions(1)
    noise, time = _utils.training_noise_and_time(rng, B)
    x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
    v_t = _serve_velocity_eager(model, _BASE, x_t, time)
    from_sampling = jnp.mean(jnp.square(v_t - (noise - actions)), axis=-1)
    loss = model.compute_loss(rng, _BASE, actions)
    print(f"loss vs sampling-path loss: max |difference| {_differs(loss, from_sampling):.3e}")
    _equal(loss, from_sampling)


def test_hard_mask_sampled_actions_ignore_images_language_and_state():
    """Latents held fixed (the base observation's), the images, the language/state tokens and the state all change:
    the sampled actions are bit for bit the same. With live latents the same perturbation moves them."""
    model = _stage_model("stage2")[1]
    cache, visible, offset = _sampling_prefix(0)
    reference = _denoise(model, 0, cache, visible, offset)
    other_obs = _utils.make_observation(7)
    other = model._lit_prefix_pass(_pre(other_obs))
    other_cache, other_visible = model._lit_extend_cache(other.kv_cache, other.visible, other.extra)
    assert not np.array_equal(np.asarray(other_cache[0][:, :, :-_LATENTS]), np.asarray(cache[0][:, :, :-_LATENTS]))
    assert not np.array_equal(np.asarray(other_cache[0][:, :, -_LATENTS:]), np.asarray(cache[0][:, :, -_LATENTS:]))
    _equal(other_visible, visible)
    _equal(other.offset, offset)
    fixed = _denoise(model, 0, _with_latents_of(other_cache, cache), other_visible, other.offset, other_obs)
    _equal(fixed, reference)
    live = _denoise(model, 0, other_cache, other_visible, other.offset, other_obs)
    print(f"live latents move the sampled actions by {_differs(live, reference):.3e}")
    assert _differs(live, reference) > 0.0


def test_with_the_masks_off_the_action_rows_do_read_the_prefix(stage2):
    with _switches(stage2, image=False, language=False):
        base = stage2._lit_prefix_pass(_pre(_sample_obs(0)))
        other_obs = _utils.make_observation(7)
        other = stage2._lit_prefix_pass(_pre(other_obs))
        extended = stage2._lit_extend_cache
        cache, visible = extended(base.kv_cache, base.visible, base.extra)
        other_cache, other_visible = extended(other.kv_cache, other.visible, base.extra)  # the same latents
        reference = _denoise(stage2, 0, cache, visible, base.offset)
        moved = _denoise(stage2, 0, other_cache, other_visible, other.offset, other_obs)
    assert _differs(moved, reference) > 0.0


def _sampled_actions(model, observations, noise):
    return [np.asarray(_sample(model, observation, noise)) for observation in observations]


def test_hard_mask_sampled_actions_are_exactly_independent_of_the_valid_token_count(monkeypatch):
    model = _stage_model("stage2")[1]
    base_obs, noise = _sample_obs(0), _sample_noise(0)
    variants = _count_variants(base_obs)
    with _constant_latents(model):
        reference = np.asarray(_sample(model, base_obs, noise))
        for name, observation in variants.items():
            _equal(_sample(model, observation, noise), reference)
            print(f"sampled actions exactly equal for: {name}")
        # the probe has power: with the raw count as the start the same samplers do differ
        monkeypatch.setattr(_pi0.Pi0, "_lit_suffix_offset", lambda self, count: count)
        raw_reference = np.asarray(_sample(model, base_obs, noise))
        leaked = {name: _differs(_sample(model, o, noise), raw_reference) for name, o in variants.items()}
    print(f"sampling from the raw count would move the actions by {leaked}")
    assert all(d > 0.0 for d in leaked.values())


def test_a_camera_missing_for_the_whole_batch_is_invisible_to_the_sampler():
    """Single-arm (left_real style): a camera is off for every sample. Its pixels reach neither the latents nor the
    action rows, even with live latents."""
    model = _stage_model("stage2")[1]
    off = tuple(("right_wrist_0_rgb", i) for i in range(B))
    observation = _sample_obs(0, masked=off)
    noisy = observation.replace(
        images={
            **observation.images,
            "right_wrist_0_rgb": jnp.asarray(
                np.random.default_rng(5).uniform(-1, 1, observation.images["right_wrist_0_rgb"].shape), jnp.float32
            ),
        }
    )
    noise = _sample_noise(0)
    actions = _sample(model, observation, noise)
    assert np.isfinite(np.asarray(actions)).all()
    _equal(_sample(model, noisy, noise), actions)


def test_sampling_is_deterministic_under_jit_and_matches_eager(capsys):
    model = _stage_model("stage2")[1]
    fn = _utils.jit_sample(model)

    def run():
        return [np.asarray(fn(_SAMPLE_RNG, _sample_obs(c), num_steps=_STEPS, noise=_sample_noise(c))) for c in range(2)]

    first, second = run(), run()
    for a, b in zip(first, second, strict=True):
        _equal(a, b)
    worst = max(_differs(a, _sampled(c)) / float(np.max(np.abs(_sampled(c)))) for c, a in enumerate(first))
    with capsys.disabled():
        print(f"\njit vs eager sampling: worst relative difference {worst:.3e} (tolerance {_REL})")
    assert worst <= _REL
