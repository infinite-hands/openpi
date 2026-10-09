import dataclasses
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


# (field, its stock value) of the fields other forks add whose loss/prefix/suffix bodies a LIT config would silently
# bypass or break; the reasons are in the Pi0Config comment. A field the config does not have counts as stock.
_LIT_INCOMPATIBLE_FIELDS = (
    ("future_tokens", "off"),
    ("action_dim_weights", None),
    ("spatial_layer", None),
    ("vlash_branches", 0),
    ("state_cond", False),
    ("image_keys", _model.IMAGE_KEYS),
    ("encode_only_active_cameras", False),
)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    # Latent Interface Training. "off" is the stock model: no extra parameters, nothing else below is read.
    # "stage1": no images; the action expert reads the language/state tokens and the goal (goal K/V), the backbone is
    #   frozen. The model never embeds an image, but the Observation must still carry every camera:
    #   preprocess_observation raises "images dict missing keys" on images={}, so a stage-1 data loader supplies and
    #   decodes all cameras.
    # "stage2": the action expert is hidden from the image columns (lit_mask_image) and the language/state columns
    #   (lit_mask_language) and reads K=lit_num_latents learned latents instead. Only both switches on is the hard
    #   firewall (then the action rows also start from a constant RoPE position, not the valid-token count). One switch
    #   alone does not isolate its modality: image and language tokens attend to each other in the prefix pass, so the
    #   columns left visible carry the hidden modality too.
    # Merge hazards, rejected for lit != "off" because git gives no signal for any of them (the other branch merges
    # cleanly and the LIT config would then silently train something else). Fields from other forks and branches:
    #   future_tokens != "off" (ih/wam-future-tokens): moves the loss into compute_loss_with_future, which train.py calls
    #     when it is set, so a WAM+LIT config would train the future-token loss and never the LIT one.
    #   action_dim_weights (ih/per-dim-loss-weighting, ih/loss-weighting-vlap): weights the stock loss line, which the
    #     LIT loss reimplements, so the weights would be dropped without a trace.
    #   spatial_layer (StreamPI ih/spatial-forcing): moves the loss body into compute_losses, called when it is set.
    #   vlash_branches > 0 (ih/vlash): its branch loss replaces the loss body LIT patches.
    #   state_cond (ih/vlash): adds the state to the adaRMS conditioning in embed_suffix, a direct state path to the
    #     action rows that breaks the stage-2 premise that they see nothing but the latents.
    #   non-default image_keys / encode_only_active_cameras (ih/vlash, left-real masking): the model would encode a
    #     camera subset while prefix_roles reads the layout from obs.images.
    lit: Literal["off", "stage1", "stage2"] = "off"
    lit_num_latents: int = 100
    lit_dim: int = 768
    # Parameter groups of the aggregator; each serves depth // lit_groups consecutive layers.
    lit_groups: int = 6
    lit_heads: int = 8
    # Width of a latent key/value: must equal num_kv_heads * head_dim of the backbone (256 for Gemma 2B).
    lit_kv_dim: int = 256
    # Latents the pose decoder reads (stage2) and goal tokens the goal encoder emits (stage1).
    lit_pose_tokens: int = 8
    lit_goal_tokens: int = 8
    # Weight of the pose loss in the training objective; the loss itself is returned unweighted.
    lit_pose_weight: float = 0.3
    # Stage-2 ablation switches: hide the image / the language+state columns from the action rows (see "stage2" above).
    lit_mask_image: bool = True
    lit_mask_language: bool = True
    # Indices into the (state-width) goal vector that the pose loss and goal encoder use: the driven arm's dims.
    lit_goal_dims: tuple[int, ...] = ()

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        self._validate_lit()

    def _validate_lit(self):
        if self.lit not in ("off", "stage1", "stage2"):
            raise ValueError(f"lit must be 'off', 'stage1' or 'stage2', got {self.lit!r}.")
        if self.lit == "off":
            return
        if not self.pi05:
            raise ValueError("lit requires pi05: pi0 feeds the state to the action rows as a suffix token.")
        for name, stock in _LIT_INCOMPATIBLE_FIELDS:
            value = getattr(self, name, stock)
            if (tuple(value) if isinstance(value, list) else value) != stock:
                raise ValueError(f"lit is not supported with {name}={value!r}: see the merge hazards in Pi0Config.")
        if not self.lit_goal_dims:
            raise ValueError("lit_goal_dims must name the goal dimensions when lit is not 'off'.")
        dims = tuple(self.lit_goal_dims)
        if len(set(dims)) != len(dims) or not all(isinstance(d, int) and 0 <= d < self.action_dim for d in dims):
            raise ValueError(f"lit_goal_dims must be distinct indices in [0, {self.action_dim}), got {dims}.")
        for name in ("lit_num_latents", "lit_dim", "lit_groups", "lit_heads", "lit_kv_dim", "lit_goal_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}.")
        if not 1 <= self.lit_pose_tokens <= self.lit_num_latents:
            raise ValueError(f"lit_pose_tokens must be in [1, lit_num_latents={self.lit_num_latents}].")
        if self.lit_pose_weight < 0:
            raise ValueError(f"lit_pose_weight must be >= 0, got {self.lit_pose_weight}.")
        if self.lit_dim % self.lit_heads != 0:
            raise ValueError(f"lit_dim {self.lit_dim} must be divisible by lit_heads {self.lit_heads}.")
        backbone = _gemma.get_config(self.paligemma_variant)
        if backbone.depth % self.lit_groups != 0:
            raise ValueError(f"lit_groups {self.lit_groups} must divide the backbone depth {backbone.depth}.")
        if self.lit_kv_dim != backbone.num_kv_heads * backbone.head_dim:
            raise ValueError(
                f"lit_kv_dim {self.lit_kv_dim} must equal num_kv_heads * head_dim = "
                f"{backbone.num_kv_heads * backbone.head_dim} of {self.paligemma_variant}."
            )
        if self.lit == "stage1" and ("lora" in self.paligemma_variant or "lora" in self.action_expert_variant):
            raise ValueError("lit='stage1' trains the whole action expert with a frozen backbone: no LoRA variants.")

    @override
    def load(
        self, params: at.Params, *, remove_extra_params: bool = True, zero_missing_regex: str | None = None
    ) -> "Pi0":
        if self.lit == "off" and (stranded := sorted(k for k in params if str(k).startswith("lit_"))):
            # remove_extra_params would drop them without a word and serve a LIT checkpoint as a plain stock model.
            raise ValueError(f"lit='off' cannot load a LIT checkpoint: its params hold {stranded}.")
        return super().load(params, remove_extra_params=remove_extra_params, zero_missing_regex=zero_missing_regex)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
            if self.lit != "off":
                observation_spec = observation_spec.replace(
                    lit_goal=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                    lit_goal_mask=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
                )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        if self.lit == "stage1":
            # Train the action expert (llm *_1, the action/time projections) and the lit_* goal encoder only.
            return nnx.All(
                nnx.Any(nnx_utils.PathRegex(".*llm.*"), nnx_utils.PathRegex("PaliGemma/img/.*")),
                nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),
            )
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
