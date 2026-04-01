from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from prompt2model.config import TaskType


class DetectionExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, topk: int) -> None:
        super().__init__()
        self.model = model
        self.topk = topk

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predictions = self.model([images[0]])[0]
        boxes = predictions["boxes"][: self.topk]
        scores = predictions["scores"][: self.topk]
        labels = predictions["labels"][: self.topk]
        pad = self.topk - boxes.shape[0]
        if pad > 0:
            boxes = torch.cat([boxes, torch.zeros((pad, 4), device=boxes.device, dtype=boxes.dtype)], dim=0)
            scores = torch.cat([scores, torch.zeros((pad,), device=scores.device, dtype=scores.dtype)], dim=0)
            labels = torch.cat([labels, torch.zeros((pad,), device=labels.device, dtype=labels.dtype)], dim=0)
        return boxes, scores, labels


def export_model_to_onnx(
    model: nn.Module,
    task: TaskType,
    example_input: torch.Tensor,
    output_path: str | Path,
    metadata: dict[str, Any],
    topk_detections: int = 20,
    opset: int = 17,
) -> str:
    output_path = str(output_path)
    model.eval()
    wrapper: nn.Module
    output_names: list[str]
    if task == TaskType.CLASSIFICATION:
        wrapper = model
        output_names = ["logits"]
    else:
        wrapper = DetectionExportWrapper(model, topk=topk_detections)
        output_names = ["boxes", "scores", "labels"]

    torch.onnx.export(
        wrapper.cpu(),
        example_input.cpu(),
        output_path,
        export_params=True,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes={"images": {0: "batch"}},
        opset_version=opset,
    )
    inject_metadata(output_path, metadata)
    return output_path


def inject_metadata(output_path: str | Path, metadata: dict[str, Any]) -> None:
    model = onnx.load(str(output_path))
    del model.metadata_props[:]
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
    onnx.save(model, str(output_path))


def read_metadata(output_path: str | Path) -> dict[str, str]:
    model = onnx.load(str(output_path))
    return {item.key: item.value for item in model.metadata_props}


def verify_onnx(output_path: str | Path, example_input: torch.Tensor) -> dict[str, Any]:
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    inputs = {session.get_inputs()[0].name: example_input.detach().cpu().numpy().astype(np.float32)}
    outputs = session.run(None, inputs)
    return {
        "metadata": read_metadata(output_path),
        "output_shapes": [list(np.array(output).shape) for output in outputs],
    }
