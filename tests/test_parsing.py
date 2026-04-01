from prompt2model.config import DatasetConfig, DatasetFormat, TaskType
from prompt2model.parsing import parse_prompt


def test_parse_prompt_extracts_task_labels_and_constraints() -> None:
    config = parse_prompt(
        prompt="Detect helmets and hard hats in low light CCTV footage and prioritize speed under 50 ms latency.",
        dataset=DatasetConfig(root="data/demo", format=DatasetFormat.COCO, annotation_path="data/demo/annotations.json"),
    )
    assert config.task == TaskType.DETECTION
    assert [label.name.lower() for label in config.labels] == ["helmets", "hard hats"]
    assert config.constraints.priority.value == "speed"
    assert config.constraints.target_latency_ms == 50
    assert "low_light" in config.augmentation_tags

