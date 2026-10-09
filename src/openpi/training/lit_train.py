"""The LIT part of the training step (scripts/train.py): the loss with its pose term, and the gradient norms logged per
module."""

import re

import flax.nnx as nnx
import jax.numpy as jnp
import optax

from openpi.models import model as _model
from openpi.shared import array_typing as at

# Module of a parameter path, first match wins; whatever is left is the backbone (SigLIP and the PaliGemma language
# model). The expert is the Gemma action expert (llm *_1) and the projections around it.
_MODULES = (
    ("lit_aggregator", re.compile(r"lit_aggregator/.*")),
    ("lit_pose_decoder", re.compile(r"lit_pose_decoder/.*")),
    ("lit_goal_encoder", re.compile(r"lit_goal_encoder/.*")),
    ("expert", re.compile(r".*llm.*_1.*|action_in_proj/.*|action_out_proj/.*|time_mlp_.*")),
    ("backbone", re.compile(r".*")),
)


def is_lit(model: _model.BaseModel) -> bool:
    return getattr(model, "lit", "off") != "off"


def loss_with_parts(
    model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
    """(the scalar loss to differentiate, the scalars to log beside it).

    lit="off" is the stock loss and logs nothing more. Otherwise the model's `compute_loss_and_aux` gives the action
    loss and the unweighted diagnostics, and the pose loss (stage 2; stage 1 has none) enters the total here, once, as
    `model.lit_pose_weight * pose_loss`: the model returns it unweighted and adds nothing itself."""
    if not is_lit(model):
        return jnp.mean(model.compute_loss(rng, observation, actions, train=True)), {}
    chunked_loss, aux = model.compute_loss_and_aux(rng, observation, actions, train=True)
    action_loss = jnp.mean(chunked_loss)
    parts = {"action_loss": action_loss, **aux}
    if model.lit == "stage2":
        if "pose_loss" not in aux:
            raise ValueError(
                "lit='stage2' trains with the pose loss, but compute_loss_and_aux returned none: the pose path is not "
                "wired, and the run would silently train the action loss alone."
            )
        return action_loss + model.lit_pose_weight * aux["pose_loss"], parts
    return action_loss, parts


def module_grad_norms(grads: nnx.State) -> dict[str, at.Array]:
    """`grad_norm_<module>` for every module that has a trainable leaf: backbone, expert and the lit_* modules."""
    leaves: dict[str, list] = {}
    for path, variable in grads.flat_state().items():
        name = next(name for name, pattern in _MODULES if pattern.fullmatch("/".join(map(str, path))))
        leaves.setdefault(name, []).append(variable.value)
    return {f"grad_norm_{name}": optax.global_norm(leaves[name]) for name, _ in _MODULES if name in leaves}


def lit_info(model: _model.BaseModel, parts: dict[str, at.Array], grads: nnx.State) -> dict[str, at.Array]:
    """The extra entries of the Step line: nothing for lit="off"."""
    if not is_lit(model):
        return {}
    return {**parts, **module_grad_norms(grads)}
