from openpi.training import config
from openpi.training.misc import ih_yam_config


def test_yam_configs_are_registered_with_expected_training_shapes():
    expected = {
        "pi05_yam_lora": "local/yam_pen_uncap",
        "pi05_yam_bolts": "local/yam_bolts",
        "pi05_yam_bagging": "local/yam_bagging_three",
        "pi05_yam_bagging_two": "local/yam_bagging_two",
        "pi05_yam_part_adapt": ih_yam_config.PART_ADAPT_REPO_ID,
        "pi05_yam_part_adapt_full": ih_yam_config.PART_ADAPT_REPO_ID,
        "pi05_yam_bagging_three": "local/yam_bagging_three",
    }

    for name, repo_id in expected.items():
        train_config = config.get_config(name)
        assert train_config.data.repo_id == repo_id
        assert train_config.model.action_horizon == 30
        assert train_config.batch_size == 64
        assert train_config.ema_decay is None

    part_adapt = config.get_config("pi05_yam_part_adapt")
    assert part_adapt.num_train_steps == ih_yam_config.PART_ADAPT_STEPS
    assert part_adapt.data.assets.assets_dir == ih_yam_config.PART_ADAPT_BASE_ASSETS_DIR
    assert part_adapt.data.assets.asset_id == ih_yam_config.PART_ADAPT_BASE_ASSET
    assert part_adapt.weight_loader.params_path == ih_yam_config.PART_ADAPT_PARAMS
