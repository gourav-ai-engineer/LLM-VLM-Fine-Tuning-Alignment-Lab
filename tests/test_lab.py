from lab.config import ExperimentConfig
from lab.data import validate_instruction_rows
from lab.metrics import mean, parameter_efficiency


def test_config_validation_and_serialization(tmp_path):
    config = ExperimentConfig(name="smoke", model_id="toy-model", dataset_id="toy-data")
    config.validate()
    output = tmp_path / "config.json"
    config.to_json(output)
    assert '"name": "smoke"' in output.read_text(encoding="utf-8")


def test_dataset_report_detects_invalid_and_duplicate_rows():
    report = validate_instruction_rows([
        {"prompt": "a", "response": "b"},
        {"prompt": "a", "response": "b"},
        {"prompt": "", "response": "c"},
    ])
    assert report.total_rows == 3
    assert report.valid_rows == 1
    assert report.duplicate_rows == 1
    assert report.missing_prompt_rows == 1


def test_metrics():
    assert parameter_efficiency(5, 100) == 5.0
    assert mean([1.0, 2.0, 3.0]) == 2.0
