from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from prompt2model.augmentations import TorchVisionAugmentationBackend, build_augmentation_plan
from prompt2model.config import TaskType
from prompt2model.models import build_classification_model
from prompt2model.training import benchmark_model, select_device, train_classification_model


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "real_datasets" / "cifar10"
OUTPUT_DIR = REPO_ROOT / "data" / "report_eval" / "cifar10_benchmark"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_balanced_indices(targets: list[int], per_class: int, seed: int) -> list[int]:
    return _build_balanced_indices_from_pool(list(range(len(targets))), targets, per_class=per_class, seed=seed)


def _build_balanced_indices_from_pool(pool: list[int], targets: list[int], per_class: int, seed: int) -> list[int]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for index in pool:
        target = targets[index]
        by_class[int(target)].append(index)
    rng = random.Random(seed)
    selected = []
    for class_id in sorted(by_class):
        indices = list(by_class[class_id])
        rng.shuffle(indices)
        selected.extend(indices[:per_class])
    rng.shuffle(selected)
    return selected


class CIFARSubsetDataset(Dataset):
    def __init__(
        self,
        base: CIFAR10,
        indices: list[int],
        image_size: int = 96,
        augmentations: TorchVisionAugmentationBackend | None = None,
        blur_eval: bool = False,
        seed: int = 42,
    ) -> None:
        self.base = base
        self.indices = indices
        self.augmentations = augmentations
        self.blur_eval = blur_eval
        self.seed = seed
        self.class_names = list(base.classes)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.indices)

    def _load_image(self, local_index: int) -> Image.Image:
        base_index = self.indices[local_index]
        image, _ = self.base[base_index]
        image = image.convert("RGB")
        if self.blur_eval:
            rng = random.Random(self.seed + local_index)
            radius = rng.uniform(1.0, 1.8)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image

    def display_image(self, local_index: int) -> Image.Image:
        return self._load_image(local_index)

    def __getitem__(self, local_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        base_index = self.indices[local_index]
        image = self._load_image(local_index)
        if self.augmentations is not None:
            image, _ = self.augmentations(image, None)
        label = int(self.base.targets[base_index])
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def _evaluate_with_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[dict[str, int]]]:
    model.eval()
    predictions: list[dict[str, int]] = []
    total = 0
    correct = 0
    local_index = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu()
            for label, pred in zip(labels, preds):
                label_int = int(label.item())
                pred_int = int(pred.item())
                predictions.append({"index": local_index, "target": label_int, "prediction": pred_int})
                correct += int(label_int == pred_int)
                total += 1
                local_index += 1
    return correct / max(total, 1), predictions


def _build_train_config() -> object:
    config = type("TrainCfg", (), {})()
    config.epochs = 2
    config.learning_rate = 8e-4
    config.weight_decay = 1e-4
    config.max_steps_per_epoch = None
    return config


def _load_or_train_model(
    variant: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> torch.nn.Module:
    run_dir = OUTPUT_DIR / variant
    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        model = build_classification_model("mobilenet_v3_small", num_classes=num_classes, pretrained=True)
        train_classification_model(
            model,
            train_loader,
            val_loader,
            _build_train_config(),
            run_dir,
            device,
        )
    model = build_classification_model("mobilenet_v3_small", num_classes=num_classes, pretrained=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    return model.to(device)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = 23
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_base = CIFAR10(root=str(DATA_ROOT), train=True, download=True)
    test_base = CIFAR10(root=str(DATA_ROOT), train=False, download=True)

    train_indices = _build_balanced_indices(train_base.targets, per_class=1000, seed=seed)
    remaining = [index for index in range(len(train_base.targets)) if index not in set(train_indices)]
    val_indices = _build_balanced_indices_from_pool(remaining, train_base.targets, per_class=200, seed=seed + 1)
    test_indices = list(range(len(test_base)))

    train_fixed = CIFARSubsetDataset(train_base, train_indices, augmentations=None, seed=seed)
    train_guided = CIFARSubsetDataset(
        train_base,
        train_indices,
        augmentations=TorchVisionAugmentationBackend(
            build_augmentation_plan(["motion_blur"], TaskType.CLASSIFICATION),
            seed=seed,
        ),
        seed=seed,
    )
    val_set = CIFARSubsetDataset(train_base, val_indices, augmentations=None, seed=seed)
    test_clean = CIFARSubsetDataset(test_base, test_indices, augmentations=None, seed=seed)
    test_blur = CIFARSubsetDataset(test_base, test_indices, augmentations=None, blur_eval=True, seed=seed)

    train_fixed_loader = DataLoader(train_fixed, batch_size=128, shuffle=True, num_workers=0)
    train_guided_loader = DataLoader(train_guided, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0)
    test_clean_loader = DataLoader(test_clean, batch_size=128, shuffle=False, num_workers=0)
    test_blur_loader = DataLoader(test_blur, batch_size=128, shuffle=False, num_workers=0)

    device = select_device(TaskType.CLASSIFICATION)
    baseline_model = _load_or_train_model("fixed", train_fixed_loader, val_loader, device, len(train_base.classes))
    guided_model = _load_or_train_model("guided", train_guided_loader, val_loader, device, len(train_base.classes))

    fixed_clean_acc, _ = _evaluate_with_predictions(baseline_model, test_clean_loader, device)
    fixed_blur_acc, _ = _evaluate_with_predictions(baseline_model, test_blur_loader, device)
    guided_clean_acc, _ = _evaluate_with_predictions(guided_model, test_clean_loader, device)
    guided_blur_acc, _ = _evaluate_with_predictions(guided_model, test_blur_loader, device)
    latency = benchmark_model(baseline_model, torch.randn(1, 3, 96, 96), device=torch.device("cpu"))

    metrics = {
        "dataset": "cifar10",
        "variant": "balanced official-train subset, full official test",
        "train_images": len(train_fixed),
        "val_images": len(val_set),
        "test_images": len(test_clean),
        "device": str(device),
        "fixed_recipe": {
            "clean_accuracy": fixed_clean_acc,
            "blur_accuracy": fixed_blur_acc,
        },
        "language_guided": {
            "clean_accuracy": guided_clean_acc,
            "blur_accuracy": guided_blur_acc,
        },
        "blur_gain_pp": (guided_blur_acc - fixed_blur_acc) * 100.0,
        "classification_latency_ms_cpu": latency["latency_ms"],
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
