from lab.configs import ExperimentConfig


def test_default_config_round_trip(tmp_path):
    config = ExperimentConfig()
    path = tmp_path / "config.json"
    config.save_json(path)
    loaded = ExperimentConfig.from_json(path)
    assert loaded == config


def test_invalid_dropout_is_rejected():
    config = ExperimentConfig(lora_dropout=1.0)
    try:
        config.validate()
    except ValueError as error:
        assert "lora_dropout" in str(error)
    else:
        raise AssertionError("invalid dropout was accepted")
