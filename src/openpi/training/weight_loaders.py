import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import jax
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


def _restore_params(params_path: str) -> at.Params:
    """The checkpoint's params as numpy arrays. A path that is not a params directory raises an opaque FileNotFoundError
    for `_METADATA` deep inside orbax, often after a container has started: say what the path has to be."""
    try:
        return _model.restore_params(download.maybe_download(params_path), restore_type=np.ndarray)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{params_path} is not a params checkpoint ({error}). --weight-loader.params-path must end in "
            "<step>/params, e.g. <experiment>/<exp-name>/19999/params: not a step directory, not an experiment root."
        ) from error


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    # Full-match regex over "/"-joined paths: the model leaves the checkpoint may lack, which then keep the model's own
    # initialisation. The default is LoRA only. A LIT stage-2 config opts in to LIT_MISSING_REGEX; anything else
    # missing from the checkpoint is an error.
    missing_regex: str = ".*lora.*"

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _restore_params(self.params_path)
        _refuse_unwanted_lit(loaded_params, params)
        # Add all missing LoRA weights (and whatever else missing_regex names).
        return _merge_params(loaded_params, params, missing_regex=self.missing_regex)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


# What a LIT stage-2 config may lack in its checkpoint: the LoRA weights, and every lit_* module (a stage-1 checkpoint
# or pi05_base has none). Anchored on purpose: `.*lit_.*` would also match any other path that merely contains "lit_".
LIT_MISSING_REGEX = ".*lora.*|lit_.*"


def _lit_leaves(tree: at.Params) -> set[str]:
    return {key for key in flax.traverse_util.flatten_dict(tree, sep="/") if key.startswith("lit_")}


def _refuse_unwanted_lit(loaded_params: at.Params, params: at.Params) -> None:
    """The generic loader's merge keeps only the checkpoint leaves the model has: a lit_* leaf the model lacks would be
    dropped without a word. A stock model (lit="off") loaded from a LIT checkpoint would train on weights that were
    trained to work with the dropped modules, and a stage-2 model given a stage-1 checkpoint would lose the goal encoder
    (the hand-off of that is LitStage1WeightLoader, which checks what it does). Refuse. (The serving path has the same
    refusal for lit="off" in Pi0Config.load.)"""
    if stray := sorted(_lit_leaves(loaded_params) - _lit_leaves(params)):
        roots = sorted({key.split("/")[0] for key in stray})
        raise ValueError(
            f"the checkpoint carries LIT leaves the model lacks ({roots}, e.g. {stray[:2]}): loading it would silently "
            "drop them and leave weights trained to work with them. Use a lit config of the same stage; a stage-1 "
            "checkpoint initialises a stage-2 model through LitStage1WeightLoader."
        )


@dataclasses.dataclass(frozen=True)
class LitStage1WeightLoader(WeightLoader):
    """Initialises a LIT stage-2 model from a stage-1 checkpoint (`params_path`, given at launch).

    The backbone and the action expert come from the checkpoint; the stage-1 goal encoder is dropped; the stage-2
    lit_* modules (aggregator, pose decoder) keep the model's own initialisation. It fails closed: it raises unless the
    checkpoint is a stage-1 one and the merge it gets back is exactly that (see `_stage1_handoff`)."""

    params_path: str = ""

    def load(self, params: at.Params) -> at.Params:
        if not self.params_path:
            raise ValueError("pass the stage-1 checkpoint's params directory: --weight-loader.params-path=<dir>.")
        loaded_params = _restore_params(self.params_path)
        return _stage1_handoff(loaded_params, params)


def _stage1_handoff(loaded_params: at.Params, params: at.Params) -> at.Params:
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    if not any(key.startswith("lit_goal_encoder/") for key in flat_loaded):
        raise ValueError("not a LIT stage-1 checkpoint: it has no lit_goal_encoder leaves.")
    if stage2 := [key for key in flat_loaded if key.startswith(("lit_aggregator/", "lit_pose_decoder/"))]:
        raise ValueError(f"not a LIT stage-1 checkpoint: it carries stage-2 modules, e.g. {stage2[:2]}.")
    if not any(key.startswith("lit_aggregator/") for key in flat_ref) or any(
        key.startswith("lit_goal_encoder/") for key in flat_ref
    ):
        raise ValueError("a stage-1 checkpoint initialises a lit='stage2' model, and this one is not.")

    # Every checkpoint leaf is consumed by the model or is the goal encoder: a leaf the stage-2 model lacks would be
    # dropped by the merge without a word.
    if dropped := sorted(key for key in flat_loaded if key not in flat_ref and not key.startswith("lit_goal_encoder/")):
        raise ValueError(
            f"the stage-1 checkpoint has leaves the stage-2 model lacks, e.g. {dropped[:3]}: not dropping them."
        )

    merged = _merge_params(loaded_params, params, missing_regex=LIT_MISSING_REGEX)

    # The fingerprint of the handoff, read off the result rather than assumed from the merge.
    flat = flax.traverse_util.flatten_dict(merged, sep="/")
    if survivors := [key for key in flat if key.startswith("lit_goal_encoder/")]:
        raise ValueError(f"stage-1 goal encoder leaves survived into the stage-2 model: {survivors[:3]}.")
    for key, reference in flat_ref.items():
        if key.startswith("lit_"):
            # untouched: the very object the model handed in, so the model keeps its own initialisation
            if flat.get(key) is not reference:
                raise ValueError(f"{key} was loaded from the checkpoint: stage-2 lit_* modules must start fresh.")
        elif key in flat_loaded:
            kept = flat.get(key)
            same = kept is not None and not isinstance(kept, jax.ShapeDtypeStruct)
            if not same or not np.array_equal(kept, np.asarray(flat_loaded[key]).astype(reference.dtype)):
                raise ValueError(f"{key} is not the stage-1 checkpoint's value after the merge.")
        elif "lora" not in key:
            raise ValueError(f"{key} is neither in the stage-1 checkpoint nor a LoRA leaf: the handoff is incomplete.")
    return merged


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
