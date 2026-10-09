"""Infinite Hands YAM training configurations."""

import dataclasses

import flax.nnx as nnx

import openpi.models.pi0_config as pi0_config
from openpi.shared import nnx_utils
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

PI05_BASE_PARAMS = "gs://openpi-assets/checkpoints/pi05_base/params"
PART_ADAPT_PARAMS = "/checkpoints/fine-tuned/pi05_yam_bagging_three/v1/19999/params"
PART_ADAPT_BASE_ASSET = "local/yam_bagging_three"
# The directory and asset id must stay paired so adaptation uses the base policy's quantiles.
PART_ADAPT_BASE_ASSETS_DIR = "/checkpoints/assets/pi05_yam_bagging_three"
PART_ADAPT_REPO_ID = "local/yam_part_adapt_current"
PART_ADAPT_STEPS = 800
# LIT (Latent Interface Training) on the pi05_yam_bagging recipe. The state/action layout is [L j0..5, L grip, R j0..5,
# R grip]; lit_goal_dims is the right arm, dims 7-13, the arm that does the task in the recipe's corpus
# (local/yam_bagging_three, 509 teleop episodes; the docs/yam_bagging_3_assets inventory of the infinite-hands repo):
# - the right arm carries a median 87.9% of the episode's joint path (p10 81.5%, min 60.7%) and every episode closes and
#   reopens its gripper;
# - the left gripper never closes (min 0.980, tracking error 0.001), so its normalised range is nearly constant and
#   the quantile normalisation amplifies its noise into the pose loss: it is excluded without exception;
# - the left joints move, but about seven times less than the right (median path 2.8 rad against 20.4 rad per episode),
#   so their goal is close to a copy of the state and would only dilute the pose loss: excluded. (The left arm is NOT
#   static in this corpus; that was measured on the 2026-08-25/26 clips, which are another set.)
# The pose-floor measurement (right arm, joints + gripper) is for exactly this set of dims.
LIT_BASE_CONFIG = "pi05_yam_bagging"
LIT_GOAL_DIMS = tuple(range(7, 14))
# All four LIT rows pin the PRODUCTION recipe's norm statistics, the ones of the run that trained the production bagging
# checkpoint (pi05_yam_bagging_three, the same assets the part-adapt rows pin), so the control and the LIT arms are
# normalised exactly like the production baseline and no row has to wait for another to write its statistics. Pinning
# the control row's own directory instead would fail every other row after container start whenever the control had
# not run first, and the launcher's compute_norm_stats pass writes to /checkpoints/assets/<config name>/<repo id>, which
# a pinned row never reads. (pi05_yam_bagging's own directory, per the adaptation notes of the infinite-hands repo, is
# a never-run config's: it is not what the production checkpoint was trained with.)
LIT_NORM_ASSETS_DIR = PART_ADAPT_BASE_ASSETS_DIR
LIT_NORM_ASSET_ID = PART_ADAPT_BASE_ASSET


def _model() -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        action_horizon=30,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )


def get_ih_yam_configs():
    # These imports are deferred because config.py expands this function while building its registry.
    from openpi.training.config import AssetsConfig
    from openpi.training.config import LeRobotAlohaDataConfig
    from openpi.training.config import TrainConfig

    def data_config(repo_id: str, prompt: str, *, assets: AssetsConfig | None = None):
        return LeRobotAlohaDataConfig(
            repo_id=repo_id,
            assets=assets or AssetsConfig(),
            default_prompt=prompt,
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        )

    def standard_config(name: str, repo_id: str, prompt: str):
        model = _model()
        return TrainConfig(
            name=name,
            model=model,
            data=data_config(repo_id, prompt),
            weight_loader=weight_loaders.CheckpointWeightLoader(PI05_BASE_PARAMS),
            batch_size=64,
            num_train_steps=20_000,
            lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=20_000),
            save_interval=1_000,
            keep_period=5_000,
            freeze_filter=model.get_freeze_filter(),
            ema_decay=None,
        )

    bagging_prompt = "place one part in the bag"

    def lit_config(suffix: str, *, lit: str, lora: bool, weight_loader, ema_decay: float | None, batch_size: int):
        """A LIT row: the pi05_yam_bagging recipe with the model switched to `lit`. Everything else (dataset, prompt,
        cameras, horizons, schedule, norm statistics) is the recipe's; `lora` False is the full fine-tune."""
        row = standard_config(f"{LIT_BASE_CONFIG}_{suffix}", "local/yam_bagging_three", bagging_prompt)
        model = dataclasses.replace(
            row.model,
            paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
            action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
            lit=lit,
            lit_goal_dims=LIT_GOAL_DIMS if lit != "off" else (),
        )
        return dataclasses.replace(
            row,
            model=model,
            data=data_config(
                "local/yam_bagging_three",
                bagging_prompt,
                assets=AssetsConfig(assets_dir=LIT_NORM_ASSETS_DIR, asset_id=LIT_NORM_ASSET_ID),
            ),
            weight_loader=weight_loader,
            batch_size=batch_size,
            freeze_filter=model.get_freeze_filter(),
            ema_decay=ema_decay,
        )

    part_model = _model()
    part_assets = AssetsConfig(
        assets_dir=PART_ADAPT_BASE_ASSETS_DIR,
        asset_id=PART_ADAPT_BASE_ASSET,
    )

    return [
        standard_config("pi05_yam_lora", "local/yam_pen_uncap", "Remove the cap from the pen"),
        standard_config("pi05_yam_bolts", "local/yam_bolts", "place one cap onto an available stud"),
        standard_config("pi05_yam_bagging", "local/yam_bagging_three", bagging_prompt),
        standard_config("pi05_yam_bagging_two", "local/yam_bagging_two", bagging_prompt),
        TrainConfig(
            name="pi05_yam_part_adapt",
            model=part_model,
            data=data_config(PART_ADAPT_REPO_ID, bagging_prompt, assets=part_assets),
            weight_loader=weight_loaders.CheckpointWeightLoader(PART_ADAPT_PARAMS),
            batch_size=64,
            num_train_steps=PART_ADAPT_STEPS,
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=50,
                peak_lr=5e-6,
                decay_steps=PART_ADAPT_STEPS,
                decay_lr=5e-7,
            ),
            save_interval=200,
            keep_period=PART_ADAPT_STEPS,
            # OpenPI names action-expert leaves with an llm _1 suffix; train only their LoRA weights.
            freeze_filter=nnx.Any(
                part_model.get_freeze_filter(),
                nnx.Not(
                    nnx.All(
                        nnx_utils.PathRegex(".*llm.*_1.*"),
                        nnx_utils.PathRegex(".*lora.*"),
                    )
                ),
            ),
            ema_decay=None,
        ),
        TrainConfig(
            name="pi05_yam_part_adapt_full",
            model=part_model,
            data=data_config(PART_ADAPT_REPO_ID, bagging_prompt, assets=part_assets),
            weight_loader=weight_loaders.CheckpointWeightLoader(PART_ADAPT_PARAMS),
            batch_size=64,
            num_train_steps=PART_ADAPT_STEPS,
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=50,
                peak_lr=5e-6,
                decay_steps=PART_ADAPT_STEPS,
                decay_lr=5e-7,
            ),
            save_interval=200,
            keep_period=PART_ADAPT_STEPS,
            freeze_filter=part_model.get_freeze_filter(),
            ema_decay=None,
        ),
        # Historical checkpoints use this name; new full runs use pi05_yam_bagging.
        standard_config("pi05_yam_bagging_three", "local/yam_bagging_three", bagging_prompt),
        # LIT (Latent Interface Training) on that recipe:
        #  _litctl   the matched control: lit off, FULL fine-tune, PaliGemma init with a random expert (LIT's tested
        #            protocol). All four rows pin the production recipe's norm statistics (LIT_NORM_ASSETS_DIR), so no
        #            statistics pass is run for any of them.
        #  _lit1     stage 1: no images reach the model, the expert + the goal encoder train, the backbone is frozen.
        #            The loader still decodes all three cameras (compute_loss needs the images dict). No EMA: the
        #            backbone is frozen.
        #  _lit2     stage 2, full fine-tune, from a stage-1 checkpoint: pass --weight-loader.params-path=<the stage-1
        #            run's <step>/params directory>.
        #  _litlite  stage 2 with LoRA on the backbone and the expert and full lit_* modules, warm start from pi05_base,
        #            the recipe's own LoRA batch size: a labelled deviation (LIT's tested protocol is the full
        #            fine-tune).
        # lit_groups=6 divides the 18 layers (3 per group); the full rows run batch 32 as the other full fine-tune rows
        # of the forks do. None of them sets action_dim_weights (the recipe has none; a LIT config rejects it).
        # Optimisation is the BASE recipe's, unchanged on every row: cosine schedule, 1,000 warmup steps, peak lr
        # 2.5e-5, the default AdamW weight decay, 20,000 steps. LIT's own pi0.5 release differs (peak lr 1e-4, warmup
        # 4,000 / 5,000 steps, weight decay 0.01: as reported by the stage review, not re-read here), and neither set is
        # tuned for these rows: measure first. The step budget is a launch-time decision too: the control's steps
        # against the combined steps of lit1 + lit2 decide whether the A/B compares equal compute or equal stages.
        lit_config("litctl", lit="off", lora=False, weight_loader=weight_loaders.PaliGemmaWeightLoader(),
                   ema_decay=0.99, batch_size=32),
        lit_config("lit1", lit="stage1", lora=False, weight_loader=weight_loaders.PaliGemmaWeightLoader(),
                   ema_decay=None, batch_size=32),
        lit_config("lit2", lit="stage2", lora=False, weight_loader=weight_loaders.LitStage1WeightLoader(),
                   ema_decay=0.99, batch_size=32),
        lit_config("litlite", lit="stage2", lora=True,
                   weight_loader=weight_loaders.CheckpointWeightLoader(
                       PI05_BASE_PARAMS, missing_regex=weight_loaders.LIT_MISSING_REGEX),
                   ema_decay=None, batch_size=64),
    ]
