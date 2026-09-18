import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_imle_config
from openpi.training import config
from openpi.training.misc import ih_imle_config


def test_imle_config_is_registered_with_a_frozen_backbone():
    train_config = config.get_config("imle_yam_firsttry")
    assert train_config.data.repo_id == ih_imle_config.FIRSTTRY_REPO_ID
    assert train_config.model.action_horizon == 30
    assert train_config.model.imle_sample_factor == 2
    assert train_config.model.model_type is config.ModelType.PI05
    assert train_config.ema_decay is None


def test_sample_factor_below_two_is_refused():
    with pytest.raises(ValueError, match="at least 2"):
        pi0_imle_config.Pi0ImleConfig(imle_sample_factor=1)


def test_lora_variants_are_refused():
    with pytest.raises(ValueError, match="non-LoRA"):
        pi0_imle_config.Pi0ImleConfig(paligemma_variant="gemma_2b_lora").get_freeze_filter()


@pytest.mark.manual
def test_single_step_sampling_and_cimle_loss_run():
    model_config = pi0_imle_config.Pi0ImleConfig(action_horizon=4, max_token_len=48)
    model = model_config.create(jax.random.key(0))
    observation, actions = model_config.fake_obs(), model_config.fake_act()

    loss = model.compute_loss(jax.random.key(1), observation, actions, train=True)
    assert loss.shape == actions.shape[:-1]
    assert np.all(np.isfinite(loss))

    sampled = model.sample_actions(jax.random.key(2), observation)
    assert sampled.shape == actions.shape
    # The head is single-step: num_steps cannot change what comes out.
    other = model.sample_actions(jax.random.key(2), observation, num_steps=1)
    assert jnp.allclose(sampled, other)
