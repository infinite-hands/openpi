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


def ema_update(
    ema_params: nnx.State, new_params: nnx.State, decay: float, trainable_filter: nnx.filterlib.Filter
) -> nnx.State:
    """The exponential moving average of the trainable leaves; every other leaf of the EMA is left as it is.

    A frozen leaf never changes, and averaging it against itself is not free: frozen leaves are stored in bfloat16, and
    the pair `decay` / `1 - decay` rounds there to values that do not sum to one (0.99 and 0.01 become 0.98828 and
    0.01001, a sum of 0.99829), so every step shrinks the leaf a little, and the EMA is what gets saved as the inference
    params. A config that freezes nothing (full fine-tune) averages every leaf exactly as before.
    """
    ema_flat, new_flat = ema_params.flat_state(), new_params.flat_state()
    trainable = set(new_params.filter(trainable_filter).flat_state())
    return nnx.State.from_flat_path(
        {
            path: jax.tree.map(lambda old, new: decay * old + (1 - decay) * new, ema_flat[path], new_flat[path])
            if path in trainable
            else ema_flat[path]
            for path in new_flat
        }
    )
