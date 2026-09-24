from collections.abc import Callable
from typing import Any

from flax import nnx
from flax import struct
import jax
import optax

from openpi.models import model as _model
from openpi.shared import array_typing as at


@at.typecheck
@struct.dataclass
class TrainState:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef[_model.BaseModel]
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: nnx.State | None = None

    # --- WAM auxiliary future-prediction loss (see docs/wam_aux_loss.md) ---
    # All default to None, so a config that does not enable the aux loss produces a TrainState
    # structurally identical to before these fields existed.
    #
    # `aux_ema_params` is DELIBERATELY separate from `ema_params` above rather than reusing it.
    # The aux loss needs an EMA copy of the model to act as its target encoder, and every IH YAM
    # config sets `ema_decay=None`, so the obvious move is to switch openpi's own EMA on. That
    # would be a silent deployment change: `checkpoints.py`'s `_split_params` exports `ema_params`
    # AS the inference "params" item whenever it is non-None, so enabling it would change which
    # weights every downstream serve, validate and warm-start loads, with no error and no log line.
    # Keeping a separate copy leaves `_split_params` untouched, and lets the aux decay be tuned
    # independently of checkpoint quality -- they want different values anyway.
    aux_ema_decay: float | None = struct.field(pytree_node=False, default=None)
    aux_ema_params: nnx.State | None = None
    # The aux predictor's own parameters, its own optax state, and the small EMA-lagged projection
    # it needs -- all bundled by wam_aux and kept entirely out of the model's `trainable_filter`,
    # so pi0's own optimizer state and checkpoint contents are unchanged.
    aux: Any | None = None


@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging. Optionally, `interp_func` can be provided to convert
    the leaf values to more meaningful strings.
    """
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")
