from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prompt2model.config import PipelineConfig


def write_markdown_report(
    output_path: str | Path,
    config: PipelineConfig,
    metrics: dict[str, Any],
    training_history: list[dict[str, float]],
    export_path: str | None,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Prompt2Model Evaluation Report",
        "",
        "## Prompt",
        config.prompt,
        "",
        "## Task Summary",
        f"- Task: `{config.task.value}`",
        f"- Model: `{config.model_name}`",
        f"- Dataset root: `{config.dataset.root}`",
        f"- Augmentations: `{', '.join(config.augmentation_tags) or 'none'}`",
        "",
        "## Resolved Labels",
    ]
    for resolved in config.resolved_labels:
        lines.append(
            f"- `{resolved.requested_label}` -> `{resolved.dataset_label}` "
            f"(method={resolved.method}, score={resolved.score:.3f})"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "```json",
            json.dumps(metrics, indent=2),
            "```",
            "",
            "## Training History",
            "```json",
            json.dumps(training_history, indent=2),
            "```",
            "",
            "## Export",
            f"- ONNX artifact: `{export_path or 'not produced'}`",
            "",
        ]
    )
    output_path.write_text("\n".join(lines))
    return str(output_path)

