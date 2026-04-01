from pathlib import Path

from prompt2model.config import DatasetConfig, DatasetFormat, TrainingConfig
from prompt2model.data import create_synthetic_detection_dataset
from prompt2model.pipeline import Prompt2ModelFactory


def test_detection_pipeline_reaches_training_and_metrics(tmp_path: Path) -> None:
    dataset_root, annotation_path = create_synthetic_detection_dataset(tmp_path / "detection_data", num_images=10, image_size=96)
    factory = Prompt2ModelFactory()
    config = factory.build_config(
        prompt="Detect squares and circles in low light scenes and prioritize speed.",
        dataset=DatasetConfig(
            root=str(dataset_root),
            format=DatasetFormat.COCO,
            annotation_path=str(annotation_path),
            image_size=96,
        ),
        training_overrides=TrainingConfig(epochs=1, batch_size=1, max_steps_per_epoch=1, device="cpu"),
    )
    config.export.output_dir = str(tmp_path / "detection_run")
    config.export.export_onnx = False
    result = factory.run(config)
    assert Path(result.report_path).exists()
    assert "map@0.5" in result.metrics
