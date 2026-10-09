import logging
from typing import NamedTuple

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import lit as _lit
from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class LitPrefix(NamedTuple):
    """What the action rows read in a LIT step, from `Pi0._lit_prefix_pass`."""

    # Stacked (k, v) of the prefix pass, each (layers, b, s, kv_heads, head_dim).
    kv_cache: tuple
    # bool (b, s): the prefix columns the action rows may attend (the LIT masks applied).
    visible: jax.Array
    # int (b,): valid prefix tokens.
    count: jax.Array
    # int (b,): the RoPE position the suffix starts from, `Pi0._lit_suffix_offset(count)`: `count`, or a constant 0
    # when every prefix column is hidden from the action rows.
    offset: jax.Array
    # (k, v, visible) of the columns appended after the prefix, k and v (layers, b, e, kv_heads, head_dim): the latents
    # (stage 2) or the goal tokens (stage 1).
    extra: tuple
    # Final latents (b, num_latents, lit_dim) for the pose decoder; None in stage 1.
    latents: jax.Array | None


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

        # LIT modules come last so that lit="off" keeps the stock parameter tree and rng stream.
        self.lit = config.lit
        if config.lit != "off":
            self.lit_goal_dims = tuple(config.lit_goal_dims)
            self.lit_mask_image = config.lit_mask_image
            self.lit_mask_language = config.lit_mask_language
            self.lit_pose_weight = config.lit_pose_weight
            self.lit_kv_heads = paligemma_config.num_kv_heads
            self.lit_head_dim = paligemma_config.head_dim
            if config.lit == "stage1":
                self.lit_goal_encoder = _lit.LitGoalEncoder(
                    pose_dim=len(config.lit_goal_dims),
                    num_tokens=config.lit_goal_tokens,
                    dim=config.lit_dim,
                    kv_dim=config.lit_kv_dim,
                    rngs=rngs,
                )
            else:
                self.lit_aggregator = _lit.LitAggregator(
                    num_latents=config.lit_num_latents,
                    dim=config.lit_dim,
                    context_dim=paligemma_config.width,
                    kv_dim=config.lit_kv_dim,
                    num_heads=config.lit_heads,
                    num_groups=config.lit_groups,
                    rngs=rngs,
                    dtype=jnp.dtype(config.dtype),
                    remat=True,
                )
                self.lit_pose_decoder = _lit.LitPoseDecoder(
                    pose_dim=len(config.lit_goal_dims),
                    num_tokens=config.lit_pose_tokens,
                    dim=config.lit_dim,
                    rngs=rngs,
                )

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        if self.lit != "off":
            return self._compute_loss_lit(rng, observation, actions, train=train)[0]
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def compute_loss_and_aux(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], dict]:
        """`compute_loss` plus scalar diagnostics: for lit != "off" {"pose_loss" (stage 2 only), "pose_copy_baseline"},
        both unweighted means over the samples whose goal is real (weight: config.lit_pose_weight); {} for lit="off"."""
        if self.lit == "off":
            return self.compute_loss(rng, observation, actions, train=train), {}
        return self._compute_loss_lit(rng, observation, actions, train=train, with_aux=True)

    @at.typecheck
    def embed_prefix_text(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Stage 1 prefix: the language/state tokens alone, no images. Same 3-tuple as `embed_prefix`."""
        tokens = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        return tokens, obs.tokenized_prompt_mask, jnp.zeros(tokens.shape[1], dtype=jnp.bool_)

    @at.typecheck
    def prefix_roles(self, obs: _model.Observation, num_tokens: int) -> at.Int[at.Array, "b s"]:
        """Role of every prefix token (lit.ROLE_PAD / ROLE_IMAGE / ROLE_SEMANTIC) of the `num_tokens` the prefix holds:
        [camera 0 | camera 1 | ... | language+state], in the order embed_prefix lays them out (a prefix of the language
        alone, as stage 1 embeds, has no camera). Image tokens are valid where their camera's mask is set,
        language/state tokens where the prompt mask is."""
        if obs.tokenized_prompt_mask is None:
            raise ValueError("LIT needs the tokenized prompt.")
        b = obs.state.shape[0]
        text_len = obs.tokenized_prompt_mask.shape[1]
        per_camera, rest = divmod(num_tokens - text_len, len(obs.images))
        if rest or per_camera < 0:
            raise ValueError(f"{num_tokens} tokens do not fit {len(obs.images)} cameras + {text_len} text.")
        roles = []
        if per_camera:
            for name in obs.images:
                valid = jnp.broadcast_to(obs.image_masks[name][..., None], (b, per_camera))
                roles.append(jnp.where(valid, _lit.ROLE_IMAGE, _lit.ROLE_PAD))
        roles.append(jnp.where(obs.tokenized_prompt_mask, _lit.ROLE_SEMANTIC, _lit.ROLE_PAD))
        return jnp.concatenate(roles, axis=1).astype(jnp.int32)

    def _lit_goal(self, obs: _model.Observation):
        """(goal [b, len(lit_goal_dims)] float32 with padded rows zeroed, valid bool[b])."""
        if obs.lit_goal is None:
            raise ValueError("lit != 'off' needs observation.lit_goal.")
        if obs.lit_goal.shape[-1] != self.action_dim:
            raise ValueError(f"lit_goal must be {self.action_dim} wide like the state, got {obs.lit_goal.shape}.")
        b = obs.state.shape[0]
        if obs.lit_goal_mask is None:
            raise ValueError(
                "observation.lit_goal needs observation.lit_goal_mask (True where the goal is real): without it a "
                "padded goal would be taken as a real target and train the pose loss and the goal K/V toward zero."
            )
        valid = jnp.broadcast_to(obs.lit_goal_mask, (b,))
        goal = jnp.broadcast_to(obs.lit_goal, (b, self.action_dim))[:, jnp.asarray(self.lit_goal_dims)]
        return jnp.where(valid[:, None], goal.astype(jnp.float32), 0.0), valid

    def _lit_kv_columns(self, k, v):
        """[layers, b, e, lit_kv_dim] -> the kv-cache layout [layers, b, e, kv_heads, head_dim]."""
        shape = (*k.shape[:3], self.lit_kv_heads, self.lit_head_dim)
        return k.reshape(shape), v.reshape(shape)

    def _lit_prefix_pass(self, observation: _model.Observation) -> LitPrefix:
        """Prefix pass, then the aggregator (stage 2) or the goal encoder (stage 1). The one routine the training loss
        and sampling both build on."""
        stage1 = self.lit == "stage1"
        embed = self.embed_prefix_text if stage1 else self.embed_prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = embed(observation)
        roles = self.prefix_roles(observation, prefix_tokens.shape[1])

        attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        count = jnp.sum(prefix_mask, axis=1)
        offset = self._lit_suffix_offset(count)
        if stage1:
            _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
            goal, valid = self._lit_goal(observation)
            goal_k, goal_v = self.lit_goal_encoder.project_kv(self.lit_goal_encoder(goal))
            layers = kv_cache[0].shape[0]
            k, v = (jnp.broadcast_to(x[None], (layers, *x.shape)) for x in (goal_k, goal_v))
            extra = (*self._lit_kv_columns(k, v), jnp.broadcast_to(valid[:, None], goal_k.shape[:2]))
            return LitPrefix(kv_cache, prefix_mask, count, offset, extra, None)

        _, kv_cache, layer_inputs = self.PaliGemma.llm(
            [prefix_tokens, None], mask=attn_mask, positions=positions, return_layer_inputs=True
        )
        semantic_mask, image_mask = _lit.role_masks(roles)
        latents, keys, values = self.lit_aggregator(layer_inputs, semantic_mask, image_mask)
        extra = (*self._lit_kv_columns(keys, values), jnp.ones(keys.shape[1:3], bool))
        return LitPrefix(kv_cache, self._lit_visible(prefix_mask, roles), count, offset, extra, latents)

    def _lit_visible(self, prefix_mask, roles):
        """The valid prefix columns the action rows may attend: the image and language/state ones are hidden per the
        stage-2 switches. Only both switches on hide everything; one alone does not isolate its modality, since image
        and language tokens attend to each other in the prefix pass and the visible columns carry the hidden one."""
        visible = prefix_mask
        if self.lit_mask_image:
            visible = visible & (roles != _lit.ROLE_IMAGE)
        if self.lit_mask_language:
            visible = visible & (roles != _lit.ROLE_SEMANTIC)
        return visible

    def _lit_suffix_offset(self, prefix_count):
        """The RoPE position the action rows start from, given the valid prefix tokens `prefix_count` (b,): the one
        place the training pass and the sampling pass both take it from.

        Latent K/V carry no RoPE while the action rows' queries are rotated by absolute position, so a start that moves
        with the valid-token count (prompt length, camera masks) moves the logits of the action rows against the
        latents: the count reaches them though every column is hidden. When both stage-2 masks are on the start is
        therefore the constant 0. Whenever a prefix column is visible (stage 1, either mask off) the rows continue
        from the count, as in the stock model."""
        if self.lit == "stage2" and self.lit_mask_image and self.lit_mask_language:
            return jnp.zeros_like(prefix_count)
        return prefix_count

    @staticmethod
    def _lit_extend_cache(kv_cache, visible, extra):
        """Append `extra` = (k, v, visible) after the cache columns, k and v cast to the cache dtype."""
        extra_k, extra_v, extra_visible = extra
        kv_cache = tuple(
            jnp.concatenate([cache, e.astype(cache.dtype)], axis=2)
            for cache, e in zip(kv_cache, (extra_k, extra_v), strict=True)
        )
        return kv_cache, jnp.concatenate([visible, extra_visible], axis=1)

    def _suffix_velocity(self, observation, x_t, time, kv_cache, visible, offset, extra=None):
        """The action expert's velocity for noisy actions `x_t` at `time` (b,) over a finished prefix pass: the one
        routine shared by the training loss and sampling.

        kv_cache: stacked prefix (k, v), (layers, b, s, kv_heads, head_dim). visible: bool (b, s), the prefix columns
        every action row may attend. offset: int (b,), the RoPE position of the first action row, as
        `LitPrefix.offset` returns it (never the raw valid-token count: see `_lit_suffix_offset`). extra: optional
        (k, v, visible) to append after the prefix columns, k and v (layers, b, e, kv_heads, head_dim), visible
        bool (b, e); they are cast to the cache dtype and carry no RoPE."""
        if extra is not None:
            kv_cache, visible = self._lit_extend_cache(kv_cache, visible, extra)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        suffix_len = suffix_tokens.shape[1]
        columns = jnp.broadcast_to(visible[:, None, :], (visible.shape[0], suffix_len, visible.shape[1]))
        full_mask = jnp.concatenate([columns, make_attn_mask(suffix_mask, suffix_ar_mask)], axis=-1)
        positions = offset[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    def _lit_pose_losses(self, observation: _model.Observation, latents) -> dict:
        goal, valid = self._lit_goal(observation)
        weights = valid.astype(jnp.float32)
        denominator = jnp.maximum(jnp.sum(weights), 1.0)

        def masked_mean(squared):
            return jnp.sum(jnp.where(valid, jnp.mean(squared, axis=-1), 0.0)) / denominator

        state = observation.state[:, jnp.asarray(self.lit_goal_dims)].astype(jnp.float32)
        aux = {"pose_copy_baseline": masked_mean(jnp.square(state - goal))}
        if latents is not None:
            aux["pose_loss"] = masked_mean(jnp.square(self.lit_pose_decoder(latents) - goal))
        return aux

    def _compute_loss_lit(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        with_aux: bool = False,
    ):
        """The LIT training loss: (action loss (b, ah), aux). Two passes instead of the stock joint one: the prefix
        pass (and the aggregator), then the action rows over [prefix cache; latent or goal K/V; own block]. The rng
        stream is the stock 3-way split; nothing here consumes randomness of its own."""
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix = self._lit_prefix_pass(observation)
        v_t = self._suffix_velocity(
            observation, x_t, time, prefix.kv_cache, prefix.visible, prefix.offset, prefix.extra
        )
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        return action_loss, (self._lit_pose_losses(observation, prefix.latents) if with_aux else {})

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        if self.lit != "off":
            return self._sample_actions_lit(rng, observation, num_steps=num_steps, noise=noise)
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def _check_lit_sampling(self):
        if self.lit == "stage1":
            raise ValueError(
                "lit='stage1' is a training-only configuration (no images, goal K/V); it cannot sample. "
                "Sample from a lit='stage2' model."
            )
        if self.lit == "off":
            raise ValueError("the LIT sampler is for lit='stage2'; lit='off' samples through the stock path.")

    def _sample_actions_lit(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""],
        noise: at.Float[at.Array, "b ah ad"] | None,
    ) -> _model.Actions:
        """Plain chunk sampling for lit='stage2': the prefix pass and the aggregator run ONCE, the latent K/V are
        appended to its cache, then every denoising step runs the training loss's `_suffix_velocity` over that cache.
        There is no memory dict and no RTC: only this plain sampler is LIT-correct."""
        self._check_lit_sampling()
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        prefix = self._lit_prefix_pass(observation)
        kv_cache, visible = self._lit_extend_cache(prefix.kv_cache, prefix.visible, prefix.extra)
        return self._denoise(observation, noise, dt, kv_cache, visible, prefix.offset)

    def _denoise(self, observation, noise, dt, kv_cache, visible, offset):
        """Euler steps of `dt` (< 0) from `noise` at t=1 to 0 over a finished prefix: `kv_cache` and `visible` already
        hold the latent columns (see `_suffix_velocity` for the arguments)."""
        batch_size = noise.shape[0]

        def step(carry):
            x_t, time = carry
            v_t = self._suffix_velocity(observation, x_t, jnp.broadcast_to(time, batch_size), kv_cache, visible, offset)
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
