"""Generates the lit=off golden fixture from the UNMODIFIED openpi pin (fb9a58b2).

    PYTHONPATH=src JAX_PLATFORMS=cpu python -m openpi.models.lit_golden.gen_baseline [--out DIR]

It records what the stock Pi0 (tiny dummy variants, real SigLIP; see lit_test_utils) computes, so that the LIT
changes can be proven not to touch lit=off:

    baseline_loss.npz     compute_loss outputs, key "<case>__s<rng seed>_train<0|1>"
    baseline_actions.npz  sample_actions outputs, key "<case>__call<1|2>_actions" (two independent calls: openpi has
                          no memory dict, sample_actions returns the actions only)
    baseline_params.json  the param pytree: path, type, shape, dtype, sum, sha256 of the values of every leaf
    baseline_gemma.npz    every pre-existing call path of the scanned gemma.Module on tiny inputs (default, prefix-only,
                          suffix over a cache, collect_attention, pooling, feature steering, dropout, bfloat16): the
                          identity reference for the opt-in per-layer-input hook of gemma.py
    baseline_meta.json    seeds, configs, toolchain versions, the git state it ran on (and, under "gemma", the same for
                          baseline_gemma.npz: `--only gemma` regenerates just that part)

Every case is run on a randomized model ("rand": randomize_zero_init, so the prefix and images matter) and some also on
the raw init ("raw", prefix-blind), in eager and module_jit mode, float32 and bfloat16. The generator refuses to run
unless src/ equals the pin apart from the LIT files (lit.py, lit_*, lit_golden/), and refuses to run with the opt-in
stub image encoder, so a regenerated fixture can only ever describe the stock model.

Exact equality against these files holds on the toolchain they were made on (baseline_meta.json); fixture_unavailable_reason
says why the exact-equality tests cannot run anywhere else.
"""

import argparse
import functools
import json
import pathlib
import platform
import subprocess
import sys

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma
from openpi.models import lit_test_utils as _utils

PIN = "fb9a58b29258323611bfcacfc84cf77ab51293a3"
FIXTURE_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

MODEL_SEED = 0
RANDOMIZE_SEED = 0
ACTIONS_SEED = 1
OBSERVATION_SEEDS = (0, 1)  # sample_actions call 1 and call 2; compute_loss uses the first
NOISE_SEEDS = (0, 1)
SAMPLE_RNG_SEED = 2
NUM_STEPS = 3
LOSS_SEEDS = (0, 1, 2)

CONFIG_FIELDS = (
    "paligemma_variant",
    "action_expert_variant",
    "pi05",
    "action_dim",
    "action_horizon",
    "max_token_len",
)
# The toolchain facts exact equality depends on.
TOOLCHAIN_KEYS = ("jax", "flax", "numpy", "machine", "jax_platform")
# case name -> (dtype, "raw" | "rand", "eager" | "jit")
CASES = {
    "f32_rand_eager": ("float32", "rand", "eager"),
    "f32_rand_jit": ("float32", "rand", "jit"),
    "f32_raw_eager": ("float32", "raw", "eager"),
    "bf16_rand_eager": ("bfloat16", "rand", "eager"),
}
# case name -> [(rng seed, train)]
LOSS_RUNS = {
    "f32_rand_eager": [(s, t) for s in LOSS_SEEDS for t in (False, True)],
    "f32_rand_jit": [(0, False), (0, True)],
    "f32_raw_eager": [(0, False)],
    "bf16_rand_eager": [(0, False)],
}


@functools.cache
def get_model(dtype: str, params: str):
    model = _utils.build_model(_utils.make_tiny_config(dtype=dtype), MODEL_SEED)
    return _utils.randomize_zero_init(model, RANDOMIZE_SEED) if params == "rand" else model


def loss_key(case: str, seed: int, *, train: bool) -> str:
    return f"{case}__s{seed}_train{int(train)}"


def actions_key(case: str, call: int) -> str:
    return f"{case}__call{call}_actions"


def run_loss(case: str) -> dict[str, np.ndarray]:
    dtype, params, mode = CASES[case]
    config, model = _utils.make_tiny_config(dtype=dtype), get_model(dtype, params)
    observation = _utils.make_observation(OBSERVATION_SEEDS[0], config=config)
    actions = _utils.make_actions(ACTIONS_SEED, config=config)
    fn = _utils.jit_loss(model) if mode == "jit" else model.compute_loss
    return {
        loss_key(case, seed, train=train): np.asarray(fn(jax.random.key(seed), observation, actions, train=train))
        for seed, train in LOSS_RUNS[case]
    }


def run_sample(case: str) -> dict[str, np.ndarray]:
    """Two independent sample_actions calls (different observation and noise)."""
    dtype, params, mode = CASES[case]
    config, model = _utils.make_tiny_config(dtype=dtype), get_model(dtype, params)
    fn = _utils.jit_sample(model) if mode == "jit" else model.sample_actions
    actions = {}
    for call, (obs_seed, noise_seed) in enumerate(zip(OBSERVATION_SEEDS, NOISE_SEEDS, strict=True), start=1):
        observation = _utils.make_observation(obs_seed, config=config)
        noise = _utils.fixed_noise(noise_seed, config=config)
        out = fn(jax.random.key(SAMPLE_RNG_SEED), observation, num_steps=NUM_STEPS, noise=noise)
        actions[actions_key(case, call)] = np.asarray(out)
    return actions


# ---- the scanned Gemma on its own: the pre-existing call paths of gemma.Module, tiny dummy variants ----

GEMMA_BATCH = 2
GEMMA_PREFIX = 6
GEMMA_SUFFIX = 3
GEMMA_SEED = 0


def gemma_inputs() -> dict:
    """Deterministic inputs of the tiny two-expert Gemma (dummy variants: width 64, depth 4, adaRMS on the second)."""
    config = gemma.get_config("dummy")
    rng = np.random.default_rng(GEMMA_SEED)

    def normal(*shape):
        return jnp.asarray(rng.normal(size=shape), jnp.float32)

    b, prefix, suffix = GEMMA_BATCH, GEMMA_PREFIX, GEMMA_SUFFIX
    total = prefix + suffix
    mask = np.ones((b, total, total), bool)
    mask[:, :prefix, prefix:] = False  # the prefix does not see the suffix
    mask[1, :, prefix - 2 : prefix] = False  # sample 1: its last two prefix tokens are padding
    prefix_pool = np.ones((b, prefix), bool)
    prefix_pool[1, -2:] = False
    depth, width = config.depth, config.width
    control = {
        "direction": normal(depth, width),
        "offset": normal(depth),
        "lower": jnp.full((depth,), -0.2, jnp.float32),
        "upper": jnp.full((depth,), 0.2, jnp.float32),
        "active": jnp.asarray([True, False, True, True]),
    }
    return {
        "configs": [config, config],
        "prefix": normal(b, prefix, width),
        "suffix": normal(b, suffix, width),
        "cond": normal(b, width),
        "positions": jnp.broadcast_to(jnp.arange(total), (b, total)),
        "mask": jnp.asarray(mask),
        "prefix_positions": jnp.broadcast_to(jnp.arange(prefix), (b, prefix)),
        "prefix_mask": jnp.asarray(mask[:, :prefix, :prefix]),
        "suffix_positions": jnp.broadcast_to(jnp.arange(prefix, total), (b, suffix)),
        "suffix_mask": jnp.ones((b, suffix, total), bool),
        "prefix_pool": jnp.asarray(prefix_pool),
        "suffix_pool": jnp.asarray([[True, True, False], [True, False, True]]),
        "query_indices": jnp.asarray([[0, 2], [1, 3]], jnp.int32),
        "band": control,
        "prefix_shift": {
            **control,
            "shift": normal(depth, b),
            "token_mask": jnp.asarray(rng.random((depth, b, prefix)) < 0.6),
        },
        "suffix_shift": {
            **control,
            "shift": normal(depth, b),
            "token_mask": jnp.asarray(rng.random((depth, b, suffix)) < 0.6),
        },
    }


def gemma_module(inputs: dict, *, dtype: str = "float32", dropout: float = 0.0) -> gemma.Module:
    return gemma.Module(configs=inputs["configs"], embed_dtype=dtype, adarms=True, dropout=dropout)


def gemma_variables(inputs: dict, *, dropout: float = 0.0):
    """Initialised params with every all-zero leaf (the adaRMS modulation kernels) filled with noise, so that the
    action expert actually reads the prefix: the rule of randomize_zero_init."""
    variables = nn.Module.init(
        gemma_module(inputs, dropout=dropout),
        jax.random.key(GEMMA_SEED),
        [inputs["prefix"], inputs["suffix"]],
        inputs["positions"],
        inputs["mask"],
        [None, inputs["cond"]],
    )
    leaves, treedef = jax.tree.flatten(variables)
    keys = jax.random.split(jax.random.key(GEMMA_SEED + 1), len(leaves))
    return treedef.unflatten(
        [
            x + 0.05 * jax.random.normal(k, x.shape, x.dtype) if not bool(jnp.any(x)) else x
            for x, k in zip(leaves, keys, strict=True)
        ]
    )


def gemma_call_paths(inputs: dict | None = None, **extra_kwargs) -> dict:
    """name -> thunk running one pre-existing call path of gemma.Module on the tiny inputs. `extra_kwargs` are passed
    to every call on top of the path's own (none for the fixture; the hook test passes its opt-in flags)."""
    inputs = inputs or gemma_inputs()
    variables, dropout_variables = gemma_variables(inputs), gemma_variables(inputs, dropout=0.3)
    module, dropout_module = gemma_module(inputs), gemma_module(inputs, dropout=0.3)
    joint = ([inputs["prefix"], inputs["suffix"]], inputs["positions"], inputs["mask"], [None, inputs["cond"]])
    prefix_only = ([inputs["prefix"], None], inputs["prefix_positions"], inputs["prefix_mask"])
    suffix_only = ([None, inputs["suffix"]], inputs["suffix_positions"], inputs["suffix_mask"], [None, inputs["cond"]])

    def apply(*args, mod=module, tree=variables, **kwargs):
        return mod.apply(tree, *args, **kwargs, **extra_kwargs)

    def over_cache(**kwargs):
        _, cache = module.apply(variables, *prefix_only)
        return apply(*suffix_only, kv_cache=cache, **kwargs)

    return {
        "joint": lambda: apply(*joint),
        "joint_bf16": lambda: apply(*joint, mod=gemma_module(inputs, dtype="bfloat16")),
        "prefix_only": lambda: apply(*prefix_only),
        "suffix_over_cache": over_cache,
        "collect_attention": lambda: apply(*joint, collect_attention=True),
        "collect_attention_selected": lambda: apply(
            *joint,
            collect_attention=True,
            attention_query_indices=inputs["query_indices"],
            attention_key_count=GEMMA_PREFIX,
        ),
        "pool_prefix_expert": lambda: apply(*prefix_only, pool_mask=inputs["prefix_pool"]),
        "pool_action_expert": lambda: over_cache(pool_mask=inputs["suffix_pool"], pool_expert=1),
        "feature_band": lambda: apply(*prefix_only, pool_mask=inputs["prefix_pool"], feature_control=inputs["band"]),
        "feature_shift_token_mask": lambda: apply(
            *prefix_only, pool_mask=inputs["prefix_pool"], feature_control=inputs["prefix_shift"]
        ),
        "feature_action_expert": lambda: over_cache(
            pool_mask=inputs["suffix_pool"], pool_expert=1, feature_control=inputs["suffix_shift"]
        ),
        "attention_pool_feature": lambda: apply(
            *prefix_only, collect_attention=True, pool_mask=inputs["prefix_pool"], feature_control=inputs["band"]
        ),
        "dropout_train": lambda: apply(
            *joint, mod=dropout_module, tree=dropout_variables, deterministic=False, rngs={"dropout": jax.random.key(1)}
        ),
        "dropout_eval": lambda: apply(*joint, mod=dropout_module, tree=dropout_variables, deterministic=True),
    }


def run_gemma() -> dict[str, np.ndarray]:
    """Every array of every call path, keyed "<path>__<position in the result>"; bfloat16 is stored as the exactly
    equal float32."""
    flat = {}
    for name, thunk in gemma_call_paths().items():
        for index, leaf in enumerate(jax.tree.leaves(thunk())):
            array = np.asarray(leaf)
            flat[f"{name}__{index}"] = array.astype(np.float32) if array.dtype == jnp.bfloat16 else array
    return flat


def param_manifests() -> dict[str, dict]:
    return {dtype: _utils.param_manifest(get_model(dtype, "raw")) for dtype in ("float32", "bfloat16")}


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True, text=True).stdout


def _is_lit_file(path: str) -> bool:
    name = pathlib.PurePosixPath(path).name
    return name == "lit.py" or name.startswith("lit_") or "/lit_golden/" in path


def stock_source_state() -> dict:
    """The git state the fixture is generated on: files under src/ that differ from the pin, other than the LIT ones
    (lit.py, lit_*, lit_golden/)."""
    paths = _git("diff", "--name-only", PIN, "--", "src").split()
    paths += _git("ls-files", "--others", "--exclude-standard", "--", "src").split()
    changed = [path for path in paths if not _is_lit_file(path)]
    return {"head": _git("rev-parse", "HEAD").strip(), "pin": PIN, "files_differing_from_pin": changed}


def toolchain() -> dict:
    return {
        "jax": jax.__version__,
        "flax": flax.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "platform": platform.platform(),
        "jax_platform": jax.default_backend(),
    }


def fixture_unavailable_reason(*, needs_siglip: bool = True) -> str | None:
    """Why exact-equality comparison against the fixture is not meaningful here, or None when it is: the fixture is
    bit-exact only on the toolchain that made it, and (for what runs SigLIP) only against the real SigLIP, not the
    opt-in stub image encoder. The exact-equality tests skip with this reason, so they read as 'not run', never as
    passing. `needs_siglip=False` is for the gemma-only fixture, which the stub does not touch."""
    if needs_siglip and _utils.stub_image_encoder_enabled():
        return (
            f"not run: {_utils.STUB_ENV_VAR} is set, the stub encoder draws different numbers than the fixture's SigLIP"
        )
    recorded = json.loads((FIXTURE_DIR / "baseline_meta.json").read_text())["toolchain"]
    current = toolchain()
    differing = {k: (recorded[k], current[k]) for k in TOOLCHAIN_KEYS if recorded[k] != current[k]}
    if differing:
        return f"not run: toolchain differs from the one that made the fixture (recorded, current): {differing}"
    return None


def meta() -> dict:
    config = _utils.make_tiny_config()
    return {
        "git": stock_source_state(),
        "toolchain": toolchain(),
        "seeds": {
            "model": MODEL_SEED,
            "randomize": RANDOMIZE_SEED,
            "actions": ACTIONS_SEED,
            "observations": list(OBSERVATION_SEEDS),
            "noise": list(NOISE_SEEDS),
            "sample_rng": SAMPLE_RNG_SEED,
            "loss": list(LOSS_SEEDS),
        },
        "num_steps": NUM_STEPS,
        "cases": {name: {"dtype": d, "params": p, "mode": m} for name, (d, p, m) in CASES.items()},
        "config": {k: getattr(config, k) for k in CONFIG_FIELDS},
    }


def gemma_meta() -> dict:
    return {
        "git": stock_source_state(),
        "toolchain": toolchain(),
        "seed": GEMMA_SEED,
        "shape": {"batch": GEMMA_BATCH, "prefix": GEMMA_PREFIX, "suffix": GEMMA_SUFFIX},
        "call_paths": sorted(gemma_call_paths()),
    }


def _refuse_unless_stock() -> None:
    if _utils.stub_image_encoder_enabled():
        sys.exit(f"refusing to generate with {_utils.STUB_ENV_VAR} set: the fixture must come from the real SigLIP")
    state = stock_source_state()
    if state["files_differing_from_pin"]:
        sys.exit(f"refusing to generate: src/ differs from the pin in {state['files_differing_from_pin']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=pathlib.Path, default=FIXTURE_DIR)
    parser.add_argument("--only", choices=("models", "gemma"), help="regenerate just one part of the fixture")
    args = parser.parse_args()
    _refuse_unless_stock()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.only in (None, "models"):
        losses: dict[str, np.ndarray] = {}
        actions: dict[str, np.ndarray] = {}
        for case in CASES:
            losses.update(run_loss(case))
            actions.update(run_sample(case))
            print(f"{case}: done", flush=True)
        np.savez(args.out / "baseline_loss.npz", **losses)
        np.savez(args.out / "baseline_actions.npz", **actions)
        for name, payload in (("baseline_params.json", param_manifests()), ("baseline_meta.json", meta())):
            (args.out / name).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        assert all(np.isfinite(a).all() for a in (*losses.values(), *actions.values())), "non-finite baseline output"
        print(f"wrote {len(losses)} loss arrays, {len(actions)} action arrays to {args.out}")

    if args.only in (None, "gemma"):
        arrays = run_gemma()
        assert all(np.isfinite(a).all() for a in arrays.values()), "non-finite gemma baseline output"
        np.savez(args.out / "baseline_gemma.npz", **arrays)
        meta_path = args.out / "baseline_meta.json"
        recorded = json.loads(meta_path.read_text())
        recorded["gemma"] = gemma_meta()
        meta_path.write_text(json.dumps(recorded, indent=1, sort_keys=True) + "\n")
        print(f"wrote {len(arrays)} gemma arrays to {args.out}")


if __name__ == "__main__":
    main()
