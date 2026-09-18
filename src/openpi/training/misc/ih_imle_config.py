"""Infinite Hands IMLE-VLA training configuration (arXiv:2609.10915)."""

import openpi.models.pi0_imle_config as pi0_imle_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from openpi.training.misc import ih_yam_config

FIRSTTRY_REPO_ID = "local/yam_fullcorpus_teleop_firsttry_20260914"
IMLE_STEPS = 20_000
# Halved from the LoRA YAM configs' 64: the expert trains in full here, and every step runs
# imle_sample_factor + 1 expert passes over the one cached backbone pass. UNMEASURED -- raise it
# once a real run reports headroom.
IMLE_BATCH_SIZE = 32


def _model() -> pi0_imle_config.Pi0ImleConfig:
    return pi0_imle_config.Pi0ImleConfig(
        pi05=True,
        action_horizon=30,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        imle_sample_factor=2,
    )


def get_ih_imle_configs():
    from openpi.training.config import TrainConfig

    model = _model()
    # The paper reports no optimizer, learning rate, schedule or batch size, so the schedule below
    # is our own pi0.5 fine-tune schedule, unchanged. That is deliberate: it leaves the head as the
    # only difference between this run and the pi0.5 baseline on the same corpus.
    return [
        TrainConfig(
            name="imle_yam_firsttry",
            model=model,
            data=ih_yam_config.get_ih_yam_data_config(FIRSTTRY_REPO_ID, ih_yam_config.BAGGING_PROMPT),
            weight_loader=weight_loaders.CheckpointWeightLoader(ih_yam_config.PI05_BASE_PARAMS),
            batch_size=IMLE_BATCH_SIZE,
            num_train_steps=IMLE_STEPS,
            lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=IMLE_STEPS),
            save_interval=1_000,
            keep_period=5_000,
            freeze_filter=model.get_freeze_filter(),
            ema_decay=None,
        ),
    ]
