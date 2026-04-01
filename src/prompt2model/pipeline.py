from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt2model.augmentations import TorchVisionAugmentationBackend, build_augmentation_plan
from prompt2model.config import DatasetConfig, PipelineConfig, TaskType, TrainingConfig
from prompt2model.data import (
    build_classification_bundle,
    build_detection_bundle,
    load_dataset_labels,
)
from prompt2model.evaluation import run_classification_inference, run_detection_inference
from prompt2model.exporting import export_model_to_onnx, verify_onnx
from prompt2model.label_resolution import LabelResolver
from prompt2model.models import (
    build_classification_model,
    build_detection_model,
    recommend_model_name,
)
from prompt2model.parsing import parse_prompt
from prompt2model.reporting import write_markdown_report
from prompt2model.training import benchmark_model, select_device, train_classification_model, train_detection_model


@dataclass
class PipelineResult:
    run_dir: str
    config_path: str
    checkpoint_path: str
    report_path: str
    metrics: dict[str, Any]
    onnx_path: str | None
    onnx_verification: dict[str, Any] | None


class Prompt2ModelFactory:
    def __init__(self, label_resolver: LabelResolver | None = None) -> None:
        self.label_resolver = label_resolver or LabelResolver()

    def build_config(
        self,
        prompt: str,
        dataset: DatasetConfig,
        task_hint: TaskType | None = None,
        training_overrides: TrainingConfig | None = None,
    ) -> PipelineConfig:
        config = parse_prompt(prompt, dataset=dataset, task_hint=task_hint, training_overrides=training_overrides)
        dataset_labels = load_dataset_labels(dataset)
        config.resolved_labels = self.label_resolver.resolve(config.labels, dataset_labels)
        if not config.model_name:
            config.model_name = recommend_model_name(config.task, config.constraints.priority)
        return config

    def run(self, config: PipelineConfig) -> PipelineResult:
        run_dir = Path(config.export.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "pipeline_config.json"

        if config.task == TaskType.DETECTION and config.dataset.image_size < 320:
            config.dataset.image_size = 320
        if config.task == TaskType.DETECTION and config.training.batch_size < 2:
            config.training.batch_size = 2

        config_path.write_text(config.model_dump_json(indent=2))

        plan = build_augmentation_plan(config.augmentation_tags, config.task)
        augmentation_backend = TorchVisionAugmentationBackend(plan, seed=config.dataset.seed)
        device = select_device(config.task, requested=config.training.device)

        if config.task == TaskType.CLASSIFICATION:
            bundle = build_classification_bundle(
                config.dataset,
                batch_size=config.training.batch_size,
                augmentations=augmentation_backend,
                num_workers=config.training.num_workers,
            )
            model = build_classification_model(
                config.model_name or "mobilenet_v3_small",
                num_classes=len(bundle.class_names),
                pretrained=config.training.pretrained,
            )
            training = train_classification_model(model, bundle.train_loader, bundle.val_loader, config.training, run_dir, device)
            metrics = run_classification_inference(model.to(device), bundle.test_loader, device)
            sample_batch, _ = next(iter(bundle.test_loader))
            benchmark = benchmark_model(model, sample_batch[:1], device=device)
        else:
            bundle = build_detection_bundle(
                config.dataset,
                batch_size=config.training.batch_size,
                augmentations=augmentation_backend,
                num_workers=config.training.num_workers,
            )
            model = build_detection_model(
                config.model_name or "ssdlite320_mobilenet_v3_large",
                num_classes=len(bundle.class_names) + 1,
                pretrained=config.training.pretrained,
            )
            training = train_detection_model(model, bundle.train_loader, bundle.val_loader, config.training, run_dir, device)
            metrics = run_detection_inference(model.to(device), bundle.test_loader, device)
            sample_images, _ = next(iter(bundle.test_loader))
            benchmark = benchmark_model(model, sample_images[:1], device=device)

        metrics = {**metrics, **benchmark}

        onnx_path: str | None = None
        verification: dict[str, Any] | None = None
        if config.export.export_onnx:
            metadata = {
                "task": config.task.value,
                "prompt": config.prompt,
                "model_name": config.model_name,
                "labels": [resolved.dataset_label for resolved in config.resolved_labels],
                "image_size": config.dataset.image_size,
            }
            try:
                if config.task == TaskType.CLASSIFICATION:
                    sample_batch, _ = next(iter(bundle.test_loader))
                    example_input = sample_batch[:1]
                else:
                    sample_images, _ = next(iter(bundle.test_loader))
                    example_input = sample_images[0].unsqueeze(0)
                onnx_path = str(run_dir / "model.onnx")
                export_model_to_onnx(
                    model=model,
                    task=config.task,
                    example_input=example_input,
                    output_path=onnx_path,
                    metadata=metadata,
                    topk_detections=config.export.topk_detections,
                    opset=config.export.onnx_opset,
                )
                verification = verify_onnx(onnx_path, example_input)
            except Exception as exc:
                metrics["onnx_export_error"] = str(exc)
                onnx_path = None
                verification = None

        report_path = write_markdown_report(
            run_dir / "evaluation_report.md",
            config=config,
            metrics=metrics,
            training_history=training.history,
            export_path=onnx_path,
        )
        return PipelineResult(
            run_dir=str(run_dir),
            config_path=str(config_path),
            checkpoint_path=training.checkpoint_path,
            report_path=report_path,
            metrics=metrics,
            onnx_path=onnx_path,
            onnx_verification=verification,
        )


def run_from_prompt(
    prompt: str,
    dataset: DatasetConfig,
    output_dir: str,
    task_hint: TaskType | None = None,
    training_overrides: TrainingConfig | None = None,
    enable_clip: bool = False,
) -> PipelineResult:
    resolver = LabelResolver(enable_clip=enable_clip)
    factory = Prompt2ModelFactory(label_resolver=resolver)
    config = factory.build_config(prompt, dataset, task_hint=task_hint, training_overrides=training_overrides)
    config.export.output_dir = output_dir
    return factory.run(config)
