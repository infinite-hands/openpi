import dataclasses
import functools
import logging
import os
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.wam_aux as wam_aux
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    # Created once and closed over, for the same reason `tx` above is: it lands in static pytree
    # metadata, and `init` below runs twice (eval_shape then jit). See wam_aux.init_aux_state.
    aux_tx = optax.adam(config.aux_learning_rate) if config.aux_loss_weight > 0 else None

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        # WAM auxiliary future-prediction loss (see docs/wam_aux_loss.md). Entirely absent unless
        # enabled: `aux=None` makes the TrainState pytree structurally identical to before this
        # feature existed, and train_step's branch on it is resolved at trace time.
        aux = None
        aux_ema_params = None
        if config.aux_loss_weight > 0:
            # The aux loss exists to shape the VISION TOWER, and it reaches it only through
            # embed_prefix -> PaliGemma.img. If the config's freeze_filter covers those params, the
            # gradient is filtered out at the optimizer and the loss trains its own predictor while
            # changing nothing about the model -- a silent no-op that still produces a healthy
            # looking curve. Note LoRA alone does NOT cause this: get_freeze_filter freezes
            # `.*llm.*`, and the vision tower is a sibling at `PaliGemma.img.*`. But a config that
            # freezes everything except its adapters (e.g. pi05_yam_part_adapt) does.
            trainable_vision = params.filter(
                nnx.All(config.trainable_filter, nnx_utils.PathRegex(".*img.*"))
            ).flat_state()
            if not trainable_vision and not config.aux_allow_frozen_vision:
                raise ValueError(
                    "aux_loss_weight > 0 but no PaliGemma.img (vision tower) parameter is "
                    "trainable under this config's freeze_filter, so the auxiliary loss could not "
                    "affect the model at all. Use a config whose vision tower trains (e.g. "
                    "pi05_yam_part_adapt_full or a standard_config one) rather than one that "
                    "freezes everything but its adapters."
                )
            aux = wam_aux.init_aux_state(
                action_dim=config.model.action_dim,
                learning_rate=config.aux_learning_rate,
                rngs=nnx.Rngs(jax.random.fold_in(rng, 0x4157)),
                tx=aux_tx,
            )
            # Seeded from the initial params, and kept separate from `ema_params` so that enabling
            # this loss cannot change what a checkpoint exports for inference.
            aux_ema_params = params

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
            aux_ema_decay=config.aux_ema_decay,
            aux_ema_params=aux_ema_params,
            aux=aux,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


# The Observation.images key, NOT the LeRobot column name: AlohaInputs maps the raw
# "cam_right_wrist" column to this. The aux loss targets the right wrist specifically because it is
# the view that sees the part being grasped.
AUX_CAMERA_KEY = "right_wrist_0_rgb"


def _minimal_observation(image: at.Array, reference: _model.Observation) -> _model.Observation:
    """A single-camera Observation for the aux path.

    `embed_prefix` iterates `obs.images` and never reads `obs.state`, so one camera in gives
    exactly that camera's tokens out, with no language branch (tokenized_prompt stays None) and
    nothing to slice. `state` is reused from the real batch purely because the field is required.
    """
    if AUX_CAMERA_KEY not in reference.images:
        # Fail rather than falling back to some other camera: silently targeting the base view
        # instead of the wrist would still train, still look healthy, and be measuring the wrong
        # thing entirely.
        raise ValueError(
            f"the aux loss targets {AUX_CAMERA_KEY!r}, which this config's observation does not "
            f"have (got {sorted(reference.images)}). A config without a right-wrist camera cannot "
            "use this loss as written."
        )
    return _model.Observation(
        images={AUX_CAMERA_KEY: image},
        image_masks={AUX_CAMERA_KEY: reference.image_masks[AUX_CAMERA_KEY]},
        state=reference.state,
    )


def _vision_grad_share(grads_primary, grads_aux, weight: float) -> dict[str, at.Array]:
    """How big the (weighted) aux gradient is next to the primary one, inside the vision tower.

    Leaves are matched by path; the two gradient trees share one structure because both come from
    the same DiffState over the same model. Per-layer numbers use SigLIP's scan-stacked layout, where
    every `encoderblock` leaf carries the layer on axis 0.
    """
    primary_leaves, _ = jax.tree_util.tree_flatten_with_path(grads_primary)
    aux_leaves = jax.tree.leaves(grads_aux)
    primary_sq = aux_sq = dot = 0.0
    aux_dominant = count = 0.0
    primary_layer_sq = aux_layer_sq = 0.0
    for (path, p), a in zip(primary_leaves, aux_leaves, strict=True):
        key = jax.tree_util.keystr(path)
        if "img" not in key:
            continue
        p = p.astype(jnp.float32)
        a = weight * a.astype(jnp.float32)
        primary_sq += jnp.sum(p * p)
        aux_sq += jnp.sum(a * a)
        dot += jnp.sum(p * a)
        aux_dominant += jnp.sum(jnp.abs(a) > jnp.abs(p))
        count += p.size
        if "encoderblock" in key:
            primary_layer_sq += jnp.sum(jnp.square(p.reshape(p.shape[0], -1)), axis=1)
            aux_layer_sq += jnp.sum(jnp.square(a.reshape(a.shape[0], -1)), axis=1)
    primary_norm, aux_norm = jnp.sqrt(primary_sq), jnp.sqrt(aux_sq)
    layer_ratio = jnp.sqrt(aux_layer_sq / (primary_layer_sq + 1e-30))
    depth = layer_ratio.shape[0]
    return {
        "gshare_img_primary_norm": primary_norm,
        "gshare_img_aux_norm": aux_norm,  # already multiplied by aux_loss_weight
        "gshare_img_aux_over_primary": aux_norm / (primary_norm + 1e-30),
        "gshare_img_cos": dot / (primary_norm * aux_norm + 1e-30),
        "gshare_img_aux_dominant_frac": aux_dominant / count,
        "gshare_img_ratio_layer_first": layer_ratio[0],
        "gshare_img_ratio_layer_mid": layer_ratio[depth // 2],
        "gshare_img_ratio_layer_last": layer_ratio[depth - 1],
    }


def _train_step_with_aux(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    model: _model.BaseModel,
    train_rng: at.KeyArrayLike,
    observation: _model.Observation,
    actions: _model.Actions,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """`train_step` with the WAM auxiliary loss added. Differentiates one combined loss w.r.t. BOTH
    the backbone (filtered by trainable_filter, exactly as the stock path does) and the aux
    predictor's own parameters, then steps each with its own optimizer."""
    if observation.future_image is None:
        raise ValueError(
            "aux_loss_weight > 0 but the batch carries no future_image -- the data config did not "
            "request the extra frame. See create_torch_dataset's delta_timestamps and "
            "SplitFutureFrame in the data config."
        )
    aux = state.aux
    projection = nnx.merge(aux.projection_graphdef, aux.projection_params)
    predictor = nnx.merge(aux.predictor_graphdef, aux.predictor_params)

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        projection: wam_aux.AuxProjection,
        predictor: wam_aux.AuxFuturePredictor,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
    ):
        primary_rng, aux_rng = jax.random.split(rng)
        primary_loss = jnp.mean(model.compute_loss(primary_rng, observation, actions, train=True))

        context_tokens, _, _ = model.embed_prefix(_minimal_observation(observation.images[AUX_CAMERA_KEY], observation))
        # The target encoder: an EMA copy of the whole model, kept SEPARATE from openpi's own
        # ema_params so enabling this cannot change which weights a checkpoint exports (see
        # TrainState). It is not part of the diffed argnums, so it is detached by construction.
        target_model = nnx.merge(state.model_def, state.aux_ema_params)
        target_tokens, _, _ = target_model.embed_prefix(
            _minimal_observation(observation.future_image, observation)
        )

        per_example, per_copy, collapse = wam_aux.compute_aux_loss(
            projection,
            predictor,
            aux.projection_ema,
            aux_rng,
            context_tokens=context_tokens,
            target_tokens_raw=target_tokens,
            offset_k=config.aux_loss_offset_k,
            action_window=actions[:, : config.aux_loss_offset_k, :],
            is_pad=observation.future_is_pad,
        )
        aux_loss = jnp.mean(per_example)
        total = primary_loss + config.aux_loss_weight * aux_loss
        return total, (primary_loss, aux_loss, jnp.mean(per_copy), collapse)

    argnums = (
        nnx.DiffState(0, config.trainable_filter),
        nnx.DiffState(1, nnx.All(nnx.Param)),
        nnx.DiffState(2, nnx.All(nnx.Param)),
    )
    grad_share_info = {}
    if config.aux_log_grad_share:
        # Same loss, split in two so each term's gradient can be measured before they are summed.
        # Each returns only its own term, so jit can drop the other term's forward computation.
        def primary_only(model, projection, predictor, rng, observation, actions):
            _, parts = loss_fn(model, projection, predictor, rng, observation, actions)
            return parts[0]

        def aux_only(model, projection, predictor, rng, observation, actions):
            _, parts = loss_fn(model, projection, predictor, rng, observation, actions)
            return parts[1], (parts[2], parts[3])

        primary_loss, (grads_primary, _, _) = nnx.value_and_grad(primary_only, argnums=argnums)(
            model, projection, predictor, train_rng, observation, actions
        )
        (aux_loss, (copy_baseline, collapse)), (grads_aux, grads_proj, grads_pred) = nnx.value_and_grad(
            aux_only, argnums=argnums, has_aux=True
        )(model, projection, predictor, train_rng, observation, actions)
        weight = config.aux_loss_weight
        grads = jax.tree.map(lambda p, a: p + weight * a, grads_primary, grads_aux)
        grads_proj = jax.tree.map(lambda g: weight * g, grads_proj)
        grads_pred = jax.tree.map(lambda g: weight * g, grads_pred)
        loss = primary_loss + weight * aux_loss
        grad_share_info = _vision_grad_share(grads_primary, grads_aux, weight)
    else:
        (loss, (primary_loss, aux_loss, copy_baseline, collapse)), (grads, grads_proj, grads_pred) = (
            nnx.value_and_grad(loss_fn, argnums=argnums, has_aux=True)(
                model, projection, predictor, train_rng, observation, actions
            )
        )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        aux=wam_aux.apply_aux_update(aux, grads_proj, grads_pred),
        aux_ema_params=jax.tree.map(
            lambda old, new: state.aux_ema_decay * old + (1 - state.aux_ema_decay) * new,
            state.aux_ema_params,
            new_params,
        ),
    )
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "primary_loss": primary_loss,
        "aux_loss": aux_loss,
        # What "the future just looks like the present" already scores. aux_loss meaningfully below
        # this is the only evidence the predictor beats the identity map.
        "aux_copy_baseline": copy_baseline,
        # 0 = target directions well spread, 1 = collapsed to one (the degenerate solution where
        # the representation stops encoding change so present and future match trivially).
        "aux_collapse": collapse,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        **grad_share_info,
    }
    return new_state, info


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # WAM auxiliary future-prediction loss (see docs/wam_aux_loss.md). `state.aux is None` is part
    # of the pytree STRUCTURE, so this branch is resolved statically at trace time -- an aux-off run
    # traces exactly the code below and nothing else.
    if state.aux is not None:
        return _train_step_with_aux(config, state, model, train_rng, observation, actions)

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    compilation_cache_dir = os.environ.get("OPENPI_JAX_COMPILATION_CACHE_DIR", "~/.cache/jax")
    jax.config.update("jax_compilation_cache_dir", str(epath.Path(compilation_cache_dir).expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    interval_started = time.monotonic()
    data_started = time.monotonic()
    batch = next(data_iter)
    host_data_wait_s = time.monotonic() - data_started
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            reduced_info["wall_s_per_step"] = (time.monotonic() - interval_started) / len(infos)
            reduced_info["host_data_wait_s_per_step"] = host_data_wait_s / len(infos)
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            logging.info(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
            interval_started = time.monotonic()
            host_data_wait_s = 0.0
        data_started = time.monotonic()
        batch = next(data_iter)
        host_data_wait_s += time.monotonic() - data_started

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
