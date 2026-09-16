"""Infinite Hands YAM training configurations."""

import flax.nnx as nnx

import openpi.models.pi0_config as pi0_config
from openpi.shared import nnx_utils
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
from openpi.training.misc.ih_weighted_loss import Pi0WeightedConfig, WeightedLossConfig

PI05_BASE_PARAMS = "gs://openpi-assets/checkpoints/pi05_base/params"
PART_ADAPT_PARAMS = "/checkpoints/fine-tuned/pi05_yam_bagging_three/v1/19999/params"
PART_ADAPT_BASE_ASSET = "local/yam_bagging_three"
# The directory and asset id must stay paired so adaptation uses the base policy's quantiles.
PART_ADAPT_BASE_ASSETS_DIR = "/checkpoints/assets/pi05_yam_bagging_three"
PART_ADAPT_REPO_ID = "local/yam_part_adapt_current"
PART_ADAPT_STEPS = 800


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
        # A/B variant of pi05_yam_part_adapt with the critical-window weighted loss
        # (openpi.training.misc.ih_weighted_loss): same data, warm start, schedule and freeze
        # filter as pi05_yam_part_adapt, so a comparison against it isolates the loss. The loss
        # knobs are `--model.loss.*` overrides.
        TrainConfig(
            name="pi05_yam_part_adapt_wl",
            model=(part_model_wl := Pi0WeightedConfig(
                pi05=True,
                action_horizon=30,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                loss=WeightedLossConfig(
                    norm_stats_dir=f"{PART_ADAPT_BASE_ASSETS_DIR}/{PART_ADAPT_BASE_ASSET}"),
            )),
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
            freeze_filter=nnx.Any(
                part_model_wl.get_freeze_filter(),
                nnx.Not(
                    nnx.All(
                        nnx_utils.PathRegex(".*llm.*_1.*"),
                        nnx_utils.PathRegex(".*lora.*"),
                    )
                ),
            ),
            ema_decay=None,
        ),
        # Historical checkpoints use this name; new full runs use pi05_yam_bagging.
        standard_config("pi05_yam_bagging_three", "local/yam_bagging_three", bagging_prompt),
    ]
