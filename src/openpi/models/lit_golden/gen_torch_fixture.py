"""Generate the torch golden fixtures for src/openpi/models/lit.py.

Extracts the interface classes (_GoalSE3Encoder, _GoalPoseDecoder and the semantic-visual aggregator) from the
Apache-2.0 MolmoAct2 LIT fork's modeling_molmoact2.py by AST, so the lerobot package need not be importable, builds them at
tiny dimensions with randomised weights from a fixed seed, and writes weights, inputs and outputs with np.savez (no
pickle). Outputs are produced in float32 and, from the same weights, in float64 (the float32 rounding floor).

    python gen_torch_fixture.py --molmo-src <modeling_molmoact2.py> [--header-dir <dir with s1_header.json, s2_header.json>]

Run it with a python that has torch and numpy. See NOTICE for provenance.
"""

import argparse
import ast
import hashlib
import json
import pathlib

import numpy as np
import torch
from torch import Tensor

HERE = pathlib.Path(__file__).parent
WANTED = {
    "_GoalSE3Encoder",
    "_GoalPoseDecoder",
    "_SemanticVisualCrossAttentionBlock",
    "_SemanticVisualSelfAttentionBlock",
    "_SemanticVisualAggregatorGroup",
    "_SemanticVisualAggregator",
}
SEED = 20261007
DIMS = {
    "num_latents": 6,
    "dim": 32,
    "context_dim": 48,
    "kv_dim": 16,
    "num_heads": 4,
    "ffn_ratio": 4.0,
    "num_groups": 3,
    "num_layers": 6,
    "pose_tokens": 2,
    "pose_dim": 7,
    "goal_tokens": 3,
    "inner_dim": 24,
    "batch": 3,
    "seq": 20,
}


def load_classes(path: pathlib.Path) -> dict:
    ns = {"torch": torch, "Tensor": Tensor}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name in WANTED:
            exec(compile(ast.Module([node], []), str(path), "exec"), ns)
    missing = WANTED - set(ns)
    if missing:
        raise RuntimeError(f"classes not found in {path}: {sorted(missing)}")
    return ns


def randomize(module: torch.nn.Module, rng: np.random.RandomState) -> None:
    """Replace every parameter by a seeded draw so no bias or LayerNorm is left at its trivial init."""
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name == "queries":
                v = rng.randn(*p.shape) * 0.5
            elif p.ndim >= 2:
                v = rng.randn(*p.shape) / np.sqrt(p.shape[-1])
            elif p.ndim == 1 and name.endswith("weight"):  # LayerNorm scale
                v = 1.0 + 0.1 * rng.randn(*p.shape)
            else:
                v = 0.1 * rng.randn(*p.shape)
            p.copy_(torch.from_numpy(v.astype(np.float32)))


def make_masks(rng: np.random.RandomState, batch: int, seq: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row token roles: image / semantic / neither (padding). Every row has at least two of each of the first two."""
    sem = np.zeros((batch, seq), bool)
    img = np.zeros((batch, seq), bool)
    for b in range(batch):
        roles = rng.choice(3, size=seq, p=[0.2, 0.45, 0.35])  # 0 pad, 1 image, 2 semantic
        roles[rng.permutation(seq)[:2]] = 1
        roles[rng.permutation(seq)[:2]] = 2
        img[b], sem[b] = roles == 1, roles == 2
    return sem, img


def build(classes: dict) -> dict:
    d = DIMS
    return {
        "agg": classes["_SemanticVisualAggregator"](
            num_tokens=d["num_latents"],
            latent_dim=d["dim"],
            context_dim=d["context_dim"],
            kv_dim=d["kv_dim"],
            num_heads=d["num_heads"],
            ffn_ratio=d["ffn_ratio"],
            dropout=0.0,
            enable_self_attention=True,
            num_layer_groups=d["num_groups"],
        ),
        "pose_norm": torch.nn.LayerNorm(d["dim"]),
        "pose_decoder": classes["_GoalPoseDecoder"](d["pose_tokens"], d["dim"], d["pose_dim"], d["inner_dim"]),
        "goal_encoder": classes["_GoalSE3Encoder"](d["pose_dim"], d["goal_tokens"], d["dim"], d["inner_dim"]),
        # Plain Linear(dim -> kv_dim, no bias): the released pi0.5 Stage-1 header has no Molmo class for these.
        "goal_to_key": torch.nn.Linear(d["dim"], d["kv_dim"], bias=False),
        "goal_to_value": torch.nn.Linear(d["dim"], d["kv_dim"], bias=False),
    }


def run(classes: dict, dtype: torch.dtype, weights: dict, inputs: dict) -> dict:
    """Forward the whole interface in `dtype`; module weights are loaded from `weights` (float32 values)."""
    mods = build(classes)
    for prefix, m in mods.items():
        m.load_state_dict(
            {k[len(prefix) + 1 :]: torch.from_numpy(v) for k, v in weights.items() if k.startswith(prefix + "/")}
        )
        m.to(dtype).eval()
    agg, norm, dec, enc = mods["agg"], mods["pose_norm"], mods["pose_decoder"], mods["goal_encoder"]
    goal_to_key, goal_to_value = mods["goal_to_key"], mods["goal_to_value"]

    t = {k: torch.from_numpy(v) for k, v in inputs.items()}
    hiddens, sem, img, goal = t["layer_hiddens"].to(dtype), t["semantic_mask"], t["image_mask"], t["goal"].to(dtype)
    out = {}
    with torch.no_grad():
        num_layers = hiddens.shape[0]
        lat = agg.initial_queries(hiddens.shape[1], device="cpu", dtype=dtype)
        per_layer, keys, values, groups = [], [], [], []
        for layer in range(num_layers):
            g = agg.layer_group_index(layer, num_layers)
            lat = agg(lat, hiddens[layer], semantic_mask=sem, image_mask=img, group_idx=g)
            k, v = agg.project_kv(lat, group_idx=g)
            per_layer.append(lat), keys.append(k), values.append(v), groups.append(g)
        pose_normed = norm(lat[:, : DIMS["pose_tokens"]])
        goal_tokens = enc(goal)
        out["latents_per_layer"] = torch.stack(per_layer)
        out["keys"], out["values"] = torch.stack(keys), torch.stack(values)
        out["final_latents"] = lat
        out["pose_normed"] = pose_normed
        out["pose_pred"] = dec(pose_normed)
        out["goal_tokens"] = goal_tokens
        out["goal_key"], out["goal_value"] = goal_to_key(goal_tokens), goal_to_value(goal_tokens)
    res = {k: v.numpy() for k, v in out.items()}
    res["group_index"] = np.array(groups, np.int64)
    return res


def header_shapes(header_dir: pathlib.Path) -> dict:
    """LIT tensors (aggregator, pose, goal) of the released pi0.5 Stage 1 / Stage 2 safetensors headers."""
    names, shapes, stages = [], [], []
    for stage in (1, 2):
        header = json.loads((header_dir / f"s{stage}_header.json").read_text())
        for name, meta in sorted(header.items()):
            if name.startswith(("model.aggregator.", "model.pose_", "model.se3_encoder.", "model.goal_to_")):
                shape = meta["shape"] + [-1] * (2 - len(meta["shape"]))
                names.append(name), shapes.append(shape), stages.append(stage)
    return {"names": np.array(names), "shapes": np.array(shapes, np.int64), "stages": np.array(stages, np.int64)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molmo-src", type=pathlib.Path, required=True)
    ap.add_argument("--header-dir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    classes = load_classes(args.molmo_src)
    rng = np.random.RandomState(SEED)
    d = DIMS

    # Build once at float32 to enumerate parameter names, randomise, and keep the float32 values as THE weights.
    torch.manual_seed(SEED)
    mods = build(classes)
    weights = {}
    for prefix, m in mods.items():
        randomize(m, rng)
        weights.update({f"{prefix}/{k}": v.detach().numpy().copy() for k, v in m.state_dict().items()})

    sem, img = make_masks(rng, d["batch"], d["seq"])
    scales = 0.5 + rng.rand(d["num_layers"], 1, 1, 1)  # layer hidden states differ in scale, as a residual stream does
    inputs = {
        "layer_hiddens": (rng.randn(d["num_layers"], d["batch"], d["seq"], d["context_dim"]) * scales).astype(
            np.float32
        ),
        "semantic_mask": sem,
        "image_mask": img,
        "goal": rng.randn(d["batch"], d["pose_dim"]).astype(np.float32),
    }
    out32 = run(classes, torch.float32, weights, inputs)
    out64 = run(classes, torch.float64, weights, inputs)

    src = args.molmo_src.read_bytes()
    save = {f"w/{k}": v for k, v in weights.items()}
    save.update({f"in/{k}": v for k, v in inputs.items()})
    save.update({f"out32/{k}": v for k, v in out32.items()})
    save.update({f"out64/{k}": v for k, v in out64.items()})
    save.update({f"dims/{k}": np.array(v) for k, v in DIMS.items()})
    save["meta/source_sha256"] = np.array(hashlib.sha256(src).hexdigest())
    save["meta/torch_version"] = np.array(torch.__version__)
    save["meta/seed"] = np.array(SEED)
    np.savez(HERE / "torch_interface.npz", **save)
    print("wrote", HERE / "torch_interface.npz", len(save), "arrays")

    if args.header_dir is not None:
        np.savez(HERE / "torch_pi05_header_shapes.npz", **header_shapes(args.header_dir))
        print("wrote", HERE / "torch_pi05_header_shapes.npz")


if __name__ == "__main__":
    main()
