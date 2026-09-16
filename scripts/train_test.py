import dataclasses
import logging
import os
import pathlib

import jax
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import train


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(
    tmp_path: pathlib.Path,
    config_name: str,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    cache_dir = tmp_path / "jax-cache"
    monkeypatch.setenv("OPENPI_JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    caplog.set_level(logging.INFO)
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)
    assert jax.config.jax_compilation_cache_dir == str(cache_dir)
    assert "wall_s_per_step=" in caplog.text
    assert "host_data_wait_s_per_step=" in caplog.text

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
