from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
    mobilenet_v3_small,
)
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    ssdlite320_mobilenet_v3_large,
)

from prompt2model.config import PriorityPreset, TaskType


def recommend_model_name(task: TaskType, priority: PriorityPreset) -> str:
    if task == TaskType.CLASSIFICATION:
        if priority == PriorityPreset.SPEED:
            return "mobilenet_v3_small"
        if priority == PriorityPreset.ACCURACY:
            return "efficientnet_b0"
        return "mobilenet_v3_large"
    if priority == PriorityPreset.SPEED:
        return "ssdlite320_mobilenet_v3_large"
    return "fasterrcnn_mobilenet_v3_large_320_fpn"


def build_classification_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    if name == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if name == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"unsupported classification model: {name}")


def build_detection_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    weights_backbone = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    if name == "ssdlite320_mobilenet_v3_large":
        return ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=weights_backbone, num_classes=num_classes)
    if name == "fasterrcnn_mobilenet_v3_large_320_fpn":
        return fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
        )
    raise ValueError(f"unsupported detection model: {name}")

