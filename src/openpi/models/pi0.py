import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

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


# Which block may attend to which, for the sequence [obs | Fp | Fs | act]: obs = images + language,
# Fp = prefix future tokens, Fs = suffix future tokens, act = action tokens. Rows are queries.
_OBS, _FP, _FS, _ACT = range(4)
_FUTURE_RULES_TRAIN_ONLY = np.array(
    [
        [1, 0, 0, 0],  # obs never reads a future token, so the prefix is what inference computes
        [1, 1, 0, 0],  # Fp reads the observation and itself
        [1, 0, 1, 1],  # Fs reads the observation, itself and the noisy actions, not Fp
        [1, 0, 0, 1],  # act reads exactly what the stock model's actions read
    ],
    dtype=bool,
)
_FUTURE_RULES_VISIBLE = np.array(
    [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],  # actions also read both future-token blocks
    ],
    dtype=bool,
)


def make_future_attn_mask_and_positions(
    prefix_mask: at.Bool[at.Array, "b p"], n_fp: int, n_fs: int, n_act: int, *, visible: bool
) -> tuple[at.Bool[at.Array, "b t t"], at.Int[at.Array, "b t"]]:
    """Attention mask and RoPE positions for [obs | Fp | Fs | act] (pi05 only).

    make_attn_mask cannot express train_only mode, where later blocks must NOT see earlier ones
    (actions may not read Fp). Positions advance only on valid tokens. In train_only mode, action
    positions are exactly the stock model's, so the action pathway attends the same keys at the
    same positions as without future tokens. In visible mode, mask and positions equal
    make_attn_mask(valid, ar) and cumsum(valid) - 1 over the whole sequence.
    """
    rules = _FUTURE_RULES_VISIBLE if visible else _FUTURE_RULES_TRAIN_ONLY
    block = np.array([_OBS] * prefix_mask.shape[1] + [_FP] * n_fp + [_FS] * n_fs + [_ACT] * n_act)
    pattern = rules[block[:, None], block[None, :]]
    batch = prefix_mask.shape[0]
    valid = jnp.concatenate([prefix_mask, jnp.ones((batch, n_fp + n_fs + n_act), dtype=jnp.bool_)], axis=1)
    mask = jnp.logical_and(pattern[None], valid[:, None, :] * valid[:, :, None])

    obs_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    n_valid = jnp.sum(prefix_mask, axis=1, keepdims=True)
    fp_positions = n_valid + jnp.arange(n_fp)[None]
    if visible:
        fs_positions = n_valid + n_fp + jnp.arange(n_fs)[None]
        act_positions = n_valid + n_fp + n_fs + jnp.arange(n_act)[None]
    else:
        fs_positions = n_valid + n_act + jnp.arange(n_fs)[None]
        act_positions = n_valid + jnp.arange(n_act)[None]
    positions = jnp.concatenate([obs_positions, fp_positions, fs_positions, act_positions], axis=1)
    return mask, positions


class _FutureHead(nnx.Module):
    """Per-token MLP from a future token's final hidden state to the target embedding."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, rngs: nnx.Rngs):
        self.fc_in = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        # Not zero-initialised: the loss is a cosine, which is undefined for a zero prediction.
        self.fc_out = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(self, x: at.Array) -> at.Array:
        return self.fc_out(nnx.gelu(self.fc_in(x.astype(jnp.float32))))


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

        # WAM future tokens. Created last so the RNG stream, and therefore every existing
        # parameter's init, is identical to the stock model. Names start with `future_` and avoid
        # `img` / `llm` / `lora`: those regexes drive freezing, the weight loader and diagnostics.
        self.future_mode = config.future_tokens
        self.num_future_tokens = config.num_future_tokens
        if config.future_tokens != "off":
            init = nnx.initializers.normal(0.02)
            n = config.num_future_tokens
            # Content-free queries (I-/V-JEPA): one shared vector plus a per-position embedding, so
            # the tokens know nothing about the image until they read it through attention.
            self.future_prefix_shared = nnx.Param(init(rngs.params(), (1, 1, paligemma_config.width)))
            self.future_prefix_pos = nnx.Param(init(rngs.params(), (1, n, paligemma_config.width)))
            self.future_suffix_shared = nnx.Param(init(rngs.params(), (1, 1, action_expert_config.width)))
            self.future_suffix_pos = nnx.Param(init(rngs.params(), (1, n, action_expert_config.width)))
            # Targets are SigLIP tokens, which PaliGemma.img already projects to the VLM width.
            self.future_prefix_head = _FutureHead(
                paligemma_config.width, config.future_head_hidden, paligemma_config.width, rngs
            )
            self.future_suffix_head = _FutureHead(
                action_expert_config.width, config.future_head_hidden, paligemma_config.width, rngs
            )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

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
        self._refuse_visible_future_tokens()
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

    def compute_loss_with_future(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, "b n d"], at.Float[at.Array, "b n d"], at.Float[at.Array, " b"]]:
        """compute_loss plus the future tokens' predictions, from ONE forward pass.

        Returns (action loss, prefix-token predictions, suffix-token predictions, flow time). The
        RNG split, preprocessing, noise and time are compute_loss's verbatim, so a same-seed run
        sees the same batch, noise and augmentation with or without future tokens.
        """
        if self.future_mode == "off":
            raise ValueError("compute_loss_with_future needs a config with future_tokens enabled")
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, _ = self.embed_prefix(observation)
        suffix_tokens, _, _, adarms_cond = self.embed_suffix(observation, x_t, time)
        batch, n = prefix_tokens.shape[0], self.num_future_tokens
        future_prefix = jnp.broadcast_to(
            self.future_prefix_shared.value + self.future_prefix_pos.value, (batch, n, prefix_tokens.shape[-1])
        ).astype(prefix_tokens.dtype)
        future_suffix = jnp.broadcast_to(
            self.future_suffix_shared.value + self.future_suffix_pos.value, (batch, n, suffix_tokens.shape[-1])
        ).astype(suffix_tokens.dtype)
        attn_mask, positions = make_future_attn_mask_and_positions(
            prefix_mask, n, n, suffix_tokens.shape[1], visible=self.future_mode == "visible"
        )
        # Suffix order [Fs, act] keeps the action readout as the last action_horizon tokens.
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [jnp.concatenate([prefix_tokens, future_prefix], axis=1), jnp.concatenate([future_suffix, suffix_tokens], axis=1)],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        prefix_prediction = self.future_prefix_head(prefix_out[:, -n:])
        suffix_prediction = self.future_suffix_head(suffix_out[:, :n])
        return action_loss, prefix_prediction, suffix_prediction, time

    def _refuse_visible_future_tokens(self) -> None:
        # A "visible" model's actions read the future tokens, which the stock compute_loss and
        # sample_actions never build: they would run, drop the tokens and shift action positions,
        # and return plausible but wrong actions with no error.
        if self.future_mode == "visible":
            raise NotImplementedError(
                "future_tokens='visible' is training-only for now: sample_actions/compute_loss do not "
                "build the future tokens this model's actions attend to"
            )

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        self._refuse_visible_future_tokens()
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
