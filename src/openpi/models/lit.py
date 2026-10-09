"""Latent Interface Training (LIT) modules: the grouped latent aggregator, its per-layer K/V, the pose decoder
(Stage 2) and the goal encoder with its goal K/V (Stage 1).

Everything is plain einsum attention over explicit (in, out) kernels, so the weight layout is unambiguous. Attention
heads are the contiguous slices of the model dimension. Each parameter group is stacked along a leading `num_groups`
axis; a group is applied to `num_layers // num_groups` consecutive layers with the latents carried through.

Parameters are float32 unless `param_dtype` says otherwise. Activations run in `dtype`; LayerNorm statistics, softmax and
the pose/goal MLPs are computed in float32. Initialisation (Xavier-uniform kernels, zero biases, unit LayerNorm scale,
truncated-normal queries with std 0.02) is ours, not copied from a reference: only forward semantics are fixed by the
parity tests.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

ROLE_PAD = 0
ROLE_IMAGE = 1
ROLE_SEMANTIC = 2

_LN_EPS = 1e-5


def role_masks(roles):
    """Context masks (semantic, image), each bool[B, S], from per-token role ids int[B, S]."""
    return roles == ROLE_SEMANTIC, roles == ROLE_IMAGE


def validate_layer_groups(num_layers: int, num_groups: int) -> None:
    if num_groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}.")
    if num_layers < 1 or num_layers % num_groups != 0:
        raise ValueError(f"num_layers must be positive and divisible by num_groups, got {num_layers} and {num_groups}.")


def layer_group_index(layer_idx: int, num_layers: int, num_groups: int) -> int:
    validate_layer_groups(num_layers, num_groups)
    if not 0 <= layer_idx < num_layers:
        raise IndexError(f"layer_idx must be in [0, {num_layers}), got {layer_idx}.")
    return layer_idx // (num_layers // num_groups)


class _Dense(nnx.Module):
    def __init__(
        self, din: int, dout: int, *, stack: tuple[int, ...] = (), bias: bool = True, rngs: nnx.Rngs, param_dtype
    ):
        init = jax.nn.initializers.xavier_uniform(in_axis=-2, out_axis=-1, batch_axis=tuple(range(len(stack))))
        self.kernel = nnx.Param(init(rngs.params(), (*stack, din, dout), param_dtype))
        self.bias = nnx.Param(jnp.zeros((*stack, dout), param_dtype)) if bias else None


class _Norm(nnx.Module):
    def __init__(self, dim: int, *, stack: tuple[int, ...] = (), param_dtype):
        self.scale = nnx.Param(jnp.ones((*stack, dim), param_dtype))
        self.bias = nnx.Param(jnp.zeros((*stack, dim), param_dtype))


class _Attn(nnx.Module):
    def __init__(self, dim: int, kv_in: int, *, stack: tuple[int, ...], rngs: nnx.Rngs, param_dtype):
        self.q = _Dense(dim, dim, stack=stack, rngs=rngs, param_dtype=param_dtype)
        self.k = _Dense(kv_in, dim, stack=stack, rngs=rngs, param_dtype=param_dtype)
        self.v = _Dense(kv_in, dim, stack=stack, rngs=rngs, param_dtype=param_dtype)
        self.o = _Dense(dim, dim, stack=stack, rngs=rngs, param_dtype=param_dtype)


class _Ffn(nnx.Module):
    def __init__(self, dim: int, inner: int, *, stack: tuple[int, ...], rngs: nnx.Rngs, param_dtype):
        self.up = _Dense(dim, inner, stack=stack, rngs=rngs, param_dtype=param_dtype)
        self.down = _Dense(inner, dim, stack=stack, rngs=rngs, param_dtype=param_dtype)


class _SelfBlock(nnx.Module):
    def __init__(self, dim: int, inner: int, *, stack: tuple[int, ...], rngs: nnx.Rngs, param_dtype):
        self.norm = _Norm(dim, stack=stack, param_dtype=param_dtype)
        self.attn = _Attn(dim, dim, stack=stack, rngs=rngs, param_dtype=param_dtype)
        self.ffn_norm = _Norm(dim, stack=stack, param_dtype=param_dtype)
        self.ffn = _Ffn(dim, inner, stack=stack, rngs=rngs, param_dtype=param_dtype)


class _CrossBlock(nnx.Module):
    def __init__(self, dim: int, context_dim: int, inner: int, *, stack: tuple[int, ...], rngs: nnx.Rngs, param_dtype):
        self.query_norm = _Norm(dim, stack=stack, param_dtype=param_dtype)
        self.context_norm = _Norm(context_dim, stack=stack, param_dtype=param_dtype)
        self.attn = _Attn(dim, context_dim, stack=stack, rngs=rngs, param_dtype=param_dtype)
        self.ffn_norm = _Norm(dim, stack=stack, param_dtype=param_dtype)
        self.ffn = _Ffn(dim, inner, stack=stack, rngs=rngs, param_dtype=param_dtype)


class _Groups(nnx.Module):
    def __init__(
        self, dim: int, context_dim: int, kv_dim: int, inner: int, num_groups: int, *, rngs: nnx.Rngs, param_dtype
    ):
        stack = (num_groups,)
        self.self_block = _SelfBlock(dim, inner, stack=stack, rngs=rngs, param_dtype=param_dtype)
        self.semantic_block = _CrossBlock(dim, context_dim, inner, stack=stack, rngs=rngs, param_dtype=param_dtype)
        self.visual_block = _CrossBlock(dim, context_dim, inner, stack=stack, rngs=rngs, param_dtype=param_dtype)
        self.to_key = _Dense(dim, kv_dim, stack=stack, bias=False, rngs=rngs, param_dtype=param_dtype)
        self.to_value = _Dense(dim, kv_dim, stack=stack, bias=False, rngs=rngs, param_dtype=param_dtype)


def _dense(p, x):
    y = jnp.einsum("...i,io->...o", x, p["kernel"].astype(x.dtype))
    return y + p["bias"].astype(x.dtype) if "bias" in p else y


def _layer_norm(p, x):
    xf = x.astype(_acc(x.dtype))
    mean = xf.mean(-1, keepdims=True)
    var = jnp.square(xf - mean).mean(-1, keepdims=True)
    y = (xf - mean) * jax.lax.rsqrt(var + _LN_EPS) * p["scale"].astype(xf.dtype) + p["bias"].astype(xf.dtype)
    return y.astype(x.dtype)


def _acc(dtype):
    """Dtype of the statistics, logits and MLP heads: float32, or float64 when the activations are float64."""
    return jnp.promote_types(dtype, jnp.float32)


def _gelu(x):
    return jax.nn.gelu(x, approximate=False)


def _ffn(p, x):
    return _dense(p["down"], _gelu(_dense(p["up"], x)))


def _attend(p, q_in, kv_in, key_mask, num_heads):
    """Multi-head attention of q_in over kv_in. key_mask bool[B, S] (True = attendable) or None.

    A masked column contributes exactly zero: its logit is the dtype minimum (exp underflows to 0) and its value is
    zeroed. A row with no attendable column attends only to a dummy key with a zero value, so its attention output is
    zero (the output bias still applies) instead of the NaN a fully masked softmax gives.
    """
    q, k, v = _dense(p["q"], q_in), _dense(p["k"], kv_in), _dense(p["v"], kv_in)
    b, nq, d = q.shape
    ns, hd = k.shape[1], d // num_heads
    q, k, v = q.reshape(b, nq, num_heads, hd), k.reshape(b, ns, num_heads, hd), v.reshape(b, ns, num_heads, hd)
    acc = _acc(q.dtype)
    scores = jnp.einsum("bqhc,bshc->bhqs", q, k, preferred_element_type=acc) * hd**-0.5
    if key_mask is not None:
        scores = jnp.where(key_mask[:, None, None, :], scores, jnp.finfo(acc).min)
        v = jnp.where(key_mask[:, :, None, None], v, 0)
    probs = jax.nn.softmax(scores, axis=-1)
    if key_mask is not None:
        probs = jnp.where(key_mask.any(-1)[:, None, None, None], probs, 0.0)
    out = jnp.einsum("bhqs,bshc->bqhc", probs.astype(v.dtype), v).reshape(b, nq, d)
    return _dense(p["o"], out)


def _self_block(p, x, num_heads):
    h = _layer_norm(p["norm"], x)
    x = x + _attend(p["attn"], h, h, None, num_heads)
    return x + _ffn(p["ffn"], _layer_norm(p["ffn_norm"], x))


def _cross_block(p, x, context, mask, num_heads):
    ctx = _layer_norm(p["context_norm"], context)
    x = x + _attend(p["attn"], _layer_norm(p["query_norm"], x), ctx, mask, num_heads)
    return x + _ffn(p["ffn"], _layer_norm(p["ffn_norm"], x))


def _group_step(p, latents, hidden, semantic_mask, image_mask, num_heads):
    latents = _self_block(p["self_block"], latents, num_heads)
    latents = _cross_block(p["semantic_block"], latents, hidden, semantic_mask, num_heads)
    return _cross_block(p["visual_block"], latents, hidden, image_mask, num_heads)


def _check_masks(hidden, semantic_mask, image_mask):
    for name, m in (("semantic_mask", semantic_mask), ("image_mask", image_mask)):
        if m.shape != hidden.shape[:2]:
            raise ValueError(f"{name} must have shape {hidden.shape[:2]}, got {m.shape}.")


class LitAggregator(nnx.Module):
    """Grouped recurrent latent aggregator over per-layer hidden states of ONE token stream.

    A bank of `num_latents` learnable queries is refreshed at every layer: pre-LN self-attention + FFN over the
    latents, then cross-attention + FFN to the language/state tokens of that layer's hidden state, then
    cross-attention + FFN to its image tokens. Layer l uses group l // (num_layers // num_groups); the latents carry
    over layers and groups. Each group also owns the key/value projections that turn the refreshed latents into the
    per-layer latent K/V (no bias, no RoPE).
    """

    def __init__(
        self,
        *,
        num_latents: int = 100,
        dim: int = 768,
        context_dim: int = 2048,
        kv_dim: int = 256,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        num_groups: int = 6,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        remat: bool = False,
    ):
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}.")
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}.")
        self.num_groups = num_groups
        self.num_heads = num_heads
        self.dtype = dtype
        self.remat = remat
        # std 0.02 truncated at +-2 absolute, as torch.nn.init.trunc_normal_(std=0.02) does (+-100 sigma).
        self.queries = nnx.Param(
            0.02 * jax.random.truncated_normal(rngs.params(), -100.0, 100.0, (num_latents, dim), param_dtype)
        )
        self.groups = _Groups(
            dim, context_dim, kv_dim, max(1, round(dim * ffn_ratio)), num_groups, rngs=rngs, param_dtype=param_dtype
        )

    def validate_num_layers(self, num_layers: int) -> None:
        validate_layer_groups(num_layers, self.num_groups)

    def group_index(self, layer_idx: int, num_layers: int) -> int:
        return layer_group_index(layer_idx, num_layers, self.num_groups)

    def initial_latents(self, batch_size: int):
        return jnp.broadcast_to(self.queries.value.astype(self.dtype), (batch_size, *self.queries.value.shape))

    def group_params(self):
        return nnx.state(self.groups, nnx.Param).to_pure_dict()

    def step(self, latents, layer_hidden, semantic_mask, image_mask, group_idx):
        """One layer: refresh `latents` [B, K, D] from `layer_hidden` [B, S, W] with group `group_idx`."""
        _check_masks(layer_hidden, semantic_mask, image_mask)
        p = jax.tree.map(lambda a: a[group_idx], self.group_params())
        return _group_step(p, latents, layer_hidden.astype(latents.dtype), semantic_mask, image_mask, self.num_heads)

    def project_kv(self, latents, group_idx):
        p = jax.tree.map(lambda a: a[group_idx], self.group_params())
        return _dense(p["to_key"], latents), _dense(p["to_value"], latents)

    def __call__(self, layer_hiddens, semantic_mask, image_mask):
        """All layers at once. layer_hiddens [L, B, S, W] -> (final latents [B, K, D], keys, values [L, B, K, kv_dim])."""
        num_layers = layer_hiddens.shape[0]
        self.validate_num_layers(num_layers)
        _check_masks(layer_hiddens[0], semantic_mask, image_mask)
        grouped = layer_hiddens.reshape(self.num_groups, num_layers // self.num_groups, *layer_hiddens.shape[1:])
        heads = self.num_heads

        def group_body(latents, xs):
            p, hiddens = xs

            def layer_body(lat, hidden):
                lat = _group_step(p, lat, hidden.astype(lat.dtype), semantic_mask, image_mask, heads)
                return lat, (_dense(p["to_key"], lat), _dense(p["to_value"], lat))

            if self.remat:
                layer_body = jax.checkpoint(layer_body)
            return jax.lax.scan(layer_body, latents, hiddens)

        latents, (keys, values) = jax.lax.scan(
            group_body, self.initial_latents(layer_hiddens.shape[1]), (self.group_params(), grouped)
        )
        return latents, keys.reshape(num_layers, *keys.shape[2:]), values.reshape(num_layers, *values.shape[2:])


class LitPoseDecoder(nnx.Module):
    """LayerNorm over the first `num_tokens` final latents, flatten, then MLP to the normalised goal pose (float32 or wider)."""

    def __init__(
        self,
        *,
        pose_dim: int,
        num_tokens: int = 8,
        dim: int = 768,
        inner_dim: int = 512,
        rngs: nnx.Rngs,
        param_dtype=jnp.float32,
    ):
        self.num_tokens = num_tokens
        self.norm = _Norm(dim, param_dtype=param_dtype)
        self.fc1 = _Dense(num_tokens * dim, inner_dim, rngs=rngs, param_dtype=param_dtype)
        self.fc2 = _Dense(inner_dim, inner_dim, rngs=rngs, param_dtype=param_dtype)
        self.fc3 = _Dense(inner_dim, pose_dim, rngs=rngs, param_dtype=param_dtype)

    def __call__(self, latents):
        """latents [B, K, D] (final latents, K >= num_tokens) -> [B, pose_dim], at least float32."""
        p = nnx.state(self, nnx.Param).to_pure_dict()
        x = _layer_norm(p["norm"], latents[:, : self.num_tokens].astype(_acc(latents.dtype)))
        x = _gelu(_dense(p["fc1"], x.reshape(x.shape[0], -1)))
        return _dense(p["fc3"], _gelu(_dense(p["fc2"], x)))


class LitGoalEncoder(nnx.Module):
    """Stage 1: normalised goal pose [B, pose_dim] -> `num_tokens` goal tokens [B, num_tokens, dim], and the goal K/V.

    The key/value projections are shared by every layer, so a goal token's K/V is the same at all layers.
    """

    def __init__(
        self,
        *,
        pose_dim: int,
        num_tokens: int = 8,
        dim: int = 768,
        kv_dim: int = 256,
        inner_dim: int = 512,
        rngs: nnx.Rngs,
        param_dtype=jnp.float32,
    ):
        self.num_tokens = num_tokens
        self.dim = dim
        self.fc1 = _Dense(pose_dim, inner_dim, rngs=rngs, param_dtype=param_dtype)
        self.fc2 = _Dense(inner_dim, inner_dim, rngs=rngs, param_dtype=param_dtype)
        self.fc3 = _Dense(inner_dim, num_tokens * dim, rngs=rngs, param_dtype=param_dtype)
        self.to_key = _Dense(dim, kv_dim, bias=False, rngs=rngs, param_dtype=param_dtype)
        self.to_value = _Dense(dim, kv_dim, bias=False, rngs=rngs, param_dtype=param_dtype)

    def __call__(self, goal):
        p = nnx.state(self, nnx.Param).to_pure_dict()
        x = _gelu(_dense(p["fc1"], goal.astype(_acc(goal.dtype))))
        x = _dense(p["fc3"], _gelu(_dense(p["fc2"], x)))
        return x.reshape(goal.shape[0], self.num_tokens, self.dim)

    def project_kv(self, tokens):
        p = nnx.state(self, nnx.Param).to_pure_dict()
        return _dense(p["to_key"], tokens), _dense(p["to_value"], tokens)
