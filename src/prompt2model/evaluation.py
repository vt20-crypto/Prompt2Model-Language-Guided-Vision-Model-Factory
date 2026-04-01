from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def evaluate_classification_predictions(predictions: list[int], targets: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
    }


def run_classification_inference(model: torch.nn.Module, data_loader: Any, device: torch.device) -> dict[str, float]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for images, target in data_loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(target.tolist())
    return evaluate_classification_predictions(predictions, labels)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    top_left = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inter_wh = (bottom_right - top_left).clamp(min=0)
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]
    union = area1[:, None] + area2 - inter_area
    return inter_area / union.clamp(min=1e-6)


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for index in range(len(precisions) - 1, 0, -1):
        precisions[index - 1] = max(precisions[index - 1], precisions[index])
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1]))


def mean_average_precision(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    thresholds = iou_thresholds or [0.5 + 0.05 * step for step in range(10)]
    class_ids = sorted(
        {
            int(label.item())
            for target in targets
            for label in target["labels"]
        }
    )
    if not class_ids:
        return {"map@0.5": 0.0, "map@[0.5:0.95]": 0.0}

    threshold_scores: dict[float, list[float]] = defaultdict(list)
    for threshold in thresholds:
        for class_id in class_ids:
            detections = []
            gt_by_image: dict[int, torch.Tensor] = {}
            for image_index, (prediction, target) in enumerate(zip(predictions, targets)):
                pred_mask = prediction["labels"] == class_id
                gt_mask = target["labels"] == class_id
                pred_boxes = prediction["boxes"][pred_mask].cpu()
                pred_scores = prediction["scores"][pred_mask].cpu()
                gt_boxes = target["boxes"][gt_mask].cpu()
                gt_by_image[image_index] = gt_boxes
                for box, score in zip(pred_boxes, pred_scores):
                    detections.append((float(score.item()), image_index, box))

            total_gt = sum(boxes.shape[0] for boxes in gt_by_image.values())
            if total_gt == 0:
                continue

            detections.sort(key=lambda item: item[0], reverse=True)
            matched: dict[int, set[int]] = defaultdict(set)
            true_positive = np.zeros(len(detections))
            false_positive = np.zeros(len(detections))

            for index, (_, image_index, predicted_box) in enumerate(detections):
                gt_boxes = gt_by_image[image_index]
                if gt_boxes.numel() == 0:
                    false_positive[index] = 1
                    continue
                ious = box_iou(predicted_box.unsqueeze(0), gt_boxes).squeeze(0)
                best_iou, best_gt = torch.max(ious, dim=0)
                best_gt_index = int(best_gt.item())
                if best_iou.item() >= threshold and best_gt_index not in matched[image_index]:
                    matched[image_index].add(best_gt_index)
                    true_positive[index] = 1
                else:
                    false_positive[index] = 1

            tp_cumsum = np.cumsum(true_positive)
            fp_cumsum = np.cumsum(false_positive)
            recalls = tp_cumsum / max(total_gt, 1)
            precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-8)
            threshold_scores[threshold].append(_compute_ap(recalls, precisions))

    map50 = float(np.mean(threshold_scores.get(0.5, [0.0])))
    all_scores = [score for scores in threshold_scores.values() for score in scores]
    return {
        "map@0.5": map50,
        "map@[0.5:0.95]": float(np.mean(all_scores)) if all_scores else 0.0,
    }


def run_detection_inference(model: torch.nn.Module, data_loader: Any, device: torch.device) -> dict[str, float]:
    model.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for images, batch_targets in data_loader:
            outputs = model([image.to(device) for image in images])
            for output, target in zip(outputs, batch_targets):
                predictions.append(
                    {
                        "boxes": output["boxes"].cpu(),
                        "scores": output["scores"].cpu(),
                        "labels": output["labels"].cpu(),
                    }
                )
                targets.append(
                    {
                        "boxes": target["boxes"].cpu(),
                        "labels": target["labels"].cpu(),
                    }
                )
    return mean_average_precision(predictions, targets)

