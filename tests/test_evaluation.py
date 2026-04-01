import torch

from prompt2model.evaluation import box_iou, mean_average_precision


def test_box_iou_returns_identity_for_same_box() -> None:
    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    iou = box_iou(box, box)
    assert torch.isclose(iou[0, 0], torch.tensor(1.0))


def test_map_is_high_for_perfect_prediction() -> None:
    predictions = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "scores": torch.tensor([0.99]),
            "labels": torch.tensor([1]),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([1]),
        }
    ]
    metrics = mean_average_precision(predictions, targets)
    assert metrics["map@0.5"] > 0.99

