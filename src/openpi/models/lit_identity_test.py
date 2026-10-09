"""lit=off must be the stock model: exact equality against the fixture generated from the unmodified pin fb9a58b2.

The fixture (lit_golden/baseline_*) holds compute_loss outputs, sample_actions outputs and the param pytree's
fingerprint, all produced by lit_golden/gen_baseline.py. Every check recomputes with the same generator code on the
current source and compares bit for bit, so a LIT change that perturbs the stock path (rng stream, param init,
attention) fails here. Do not regenerate the fixture from a LIT head: the generator refuses to, and a regenerated
fixture would prove nothing.

Exact equality holds only on the toolchain the fixture was made on (arm64 CPU, jax 0.5.3, flax 0.10.2, numpy 1.26.4: see
baseline_meta.json) and only with the real SigLIP. On any other toolchain, or with the opt-in stub image encoder
(LIT_TEST_STUB_IMAGE_ENCODER), the exact-equality tests SKIP with a "not run: ..." reason naming the mismatch: they must
read as not run there, never as passing. The structure checks (names, shapes, dtypes of the param tree, the fixture's own
provenance) are platform-independent and always run.

The whole file takes a few minutes on CPU (the 27-layer SigLIP dominates). For a quicker gate during development run
`-k "fixture or lit_off or param_tree or f32_rand_jit"`, then the full file before handing off.
"""

import functools
import json

import numpy as np
import pytest

from openpi.models import lit_test_utils as _utils
from openpi.models import pi0_config
from openpi.models.lit_golden import gen_baseline as _gen

_NOT_RUN = _gen.fixture_unavailable_reason()
exact_equality = pytest.mark.skipif(_NOT_RUN is not None, reason=_NOT_RUN or "")
_NOT_RUN_GEMMA = _gen.fixture_unavailable_reason(needs_siglip=False)
exact_equality_gemma = pytest.mark.skipif(_NOT_RUN_GEMMA is not None, reason=_NOT_RUN_GEMMA or "")


@functools.cache
def _fixture(name: str):
    path = _gen.FIXTURE_DIR / name
    if name.endswith(".npz"):
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    return json.loads(path.read_text())


@functools.cache
def _loss(case: str):
    return _gen.run_loss(case)


@functools.cache
def _sample(case: str):
    return _gen.run_sample(case)


def test_fixture_was_generated_from_the_pin():
    meta = _fixture("baseline_meta.json")
    assert meta["git"]["pin"] == _gen.PIN
    assert meta["git"]["head"] == _gen.PIN
    assert meta["git"]["files_differing_from_pin"] == []
    assert set(_gen.TOOLCHAIN_KEYS) <= set(meta["toolchain"])


def test_gemma_fixture_was_generated_from_the_pin():
    meta = _fixture("baseline_meta.json")["gemma"]
    assert meta["git"]["pin"] == _gen.PIN
    assert meta["git"]["head"] == _gen.PIN
    assert meta["git"]["files_differing_from_pin"] == []
    assert set(_gen.TOOLCHAIN_KEYS) <= set(meta["toolchain"])


def test_fixture_matches_the_generator_definition():
    meta = _fixture("baseline_meta.json")
    current = _gen.meta()
    for key in ("seeds", "num_steps", "cases", "config"):
        assert meta[key] == current[key], f"{key} changed since the fixture was generated"
    gemma = meta["gemma"]
    current_gemma = _gen.gemma_meta()
    for key in ("seed", "shape", "call_paths"):
        assert gemma[key] == current_gemma[key], f"gemma {key} changed since the fixture was generated"


def test_stock_config_defaults_to_lit_off():
    config = pi0_config.Pi0Config()
    assert getattr(config, "lit", "off") == "off"
    assert getattr(_utils.make_tiny_config(), "lit", "off") == "off"


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_param_tree_structure_is_unchanged(dtype):
    expected, got = _fixture("baseline_params.json")[dtype], _utils.param_manifest(_gen.get_model(dtype, "raw"))
    if _utils.stub_image_encoder_enabled():
        expected = {p: v for p, v in expected.items() if not p.startswith("PaliGemma/img/")}
        got = {p: v for p, v in got.items() if not p.startswith("PaliGemma/img/")}
    assert sorted(got) == sorted(expected), (
        f"added {sorted(set(got) - set(expected))}, removed {sorted(set(expected) - set(got))}"
    )
    for path, want in expected.items():
        have = got[path]
        assert (have["type"], have["shape"], have["dtype"]) == (want["type"], want["shape"], want["dtype"]), path


@exact_equality
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_param_init_values_are_unchanged(dtype):
    expected, got = _fixture("baseline_params.json")[dtype], _utils.param_manifest(_gen.get_model(dtype, "raw"))
    changed = [path for path in expected if got[path]["sha256"] != expected[path]["sha256"]]
    assert not changed, f"init values changed (rng stream moved?) for {changed}"


@exact_equality
@pytest.mark.parametrize("case", list(_gen.CASES))
def test_compute_loss_is_bit_identical(case):
    expected = {k: v for k, v in _fixture("baseline_loss.npz").items() if k.startswith(f"{case}__")}
    got = _loss(case)
    assert expected
    assert sorted(got) == sorted(expected)
    for key, want in expected.items():
        assert (got[key].dtype, got[key].shape) == (want.dtype, want.shape), key
        np.testing.assert_array_equal(got[key], want, err_msg=key)


@exact_equality
@pytest.mark.parametrize("case", list(_gen.CASES))
def test_sample_actions_is_bit_identical(case):
    expected = {k: v for k, v in _fixture("baseline_actions.npz").items() if k.startswith(f"{case}__")}
    got = _sample(case)
    assert expected
    assert sorted(got) == sorted(expected)
    for key, want in expected.items():
        assert (got[key].dtype, got[key].shape) == (want.dtype, want.shape), key
        np.testing.assert_array_equal(got[key], want, err_msg=key)


@exact_equality_gemma
def test_gemma_call_paths_are_bit_identical():
    """Every pre-existing call path of the scanned gemma.Module (default, prefix-only, over a cache, collect_attention,
    pooling, feature steering, dropout, bfloat16), recomputed on the current gemma.py, equals the fixture bit for bit."""
    expected = _fixture("baseline_gemma.npz")
    got = _gen.run_gemma()
    assert sorted(got) == sorted(expected)
    for key, want in expected.items():
        assert (got[key].dtype, got[key].shape) == (want.dtype, want.shape), key
        np.testing.assert_array_equal(got[key], want, err_msg=key)
