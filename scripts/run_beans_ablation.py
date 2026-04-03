from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from prompt2model.augmentations import TorchVisionAugmentationBackend, build_augmentation_plan
from prompt2model.config import TaskType
from prompt2model.models import build_classification_model
from prompt2model.training import select_device, train_classification_model


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "report_eval" / "beans_ablation"
MAIN_BENCHMARK_DIR = REPO_ROOT / "data" / "report_eval" / "beans_benchmark"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _apply_low_light(image: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.22, 0.38))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.65, 0.9))
    image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.0)))
    array = np.asarray(image).astype(np.float32)
    noise = np.random.default_rng(seed).normal(0, 10, size=array.shape)
    array = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


class BeansDataset(Dataset):
    def __init__(
        self,
        split,
        image_size: int = 160,
        augmentations: TorchVisionAugmentationBackend | None = None,
        low_light_eval: bool = False,
        seed: int = 42,
    ) -> None:
        self.split = split
        self.augmentations = augmentations
        self.low_light_eval = low_light_eval
        self.seed = seed
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.split[index]["image"].convert("RGB")
        if self.low_light_eval:
            image = _apply_low_light(image, self.seed + index)
        if self.augmentations is not None:
            image, _ = self.augmentations(image, None)
        label = int(self.split[index]["labels"])
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def _evaluate_accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu()
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
    return correct / max(total, 1)


def _train_config(epochs: int = 4) -> object:
    config = type("TrainCfg", (), {})()
    config.epochs = epochs
    config.learning_rate = 1e-3
    config.weight_decay = 1e-4
    config.max_steps_per_epoch = None
    return config


def _run_config(
    name: str,
    model_name: str,
    pretrained: bool,
    checkpoint_override: Path | None,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_clean_loader: DataLoader,
    test_low_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, float | str]:
    run_dir = OUTPUT_DIR / name
    checkpoint_path = checkpoint_override or (run_dir / "best_model.pt")
    if not checkpoint_path.exists():
        model = build_classification_model(model_name, num_classes=num_classes, pretrained=pretrained)
        train_classification_model(
            model,
            train_loader,
            val_loader,
            _train_config(),
            run_dir,
            device,
        )
    model = build_classification_model(model_name, num_classes=num_classes, pretrained=pretrained)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model = model.to(device)
    return {
        "name": name,
        "model_name": model_name,
        "pretrained": pretrained,
        "clean_accuracy": _evaluate_accuracy(model, test_clean_loader, device),
        "low_light_accuracy": _evaluate_accuracy(model, test_low_loader, device),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = 17
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = load_dataset("beans")
    num_classes = len(dataset["train"].features["labels"].names)
    device = select_device(TaskType.CLASSIFICATION)

    train_fixed = BeansDataset(dataset["train"], augmentations=None, seed=seed)
    train_guided = BeansDataset(
        dataset["train"],
        augmentations=TorchVisionAugmentationBackend(
            build_augmentation_plan(["low_light"], TaskType.CLASSIFICATION),
            seed=seed,
        ),
        seed=seed,
    )
    val_set = BeansDataset(dataset["validation"], augmentations=None, seed=seed)
    test_clean = BeansDataset(dataset["test"], augmentations=None, seed=seed)
    test_low = BeansDataset(dataset["test"], augmentations=None, low_light_eval=True, seed=seed)

    train_fixed_loader = DataLoader(train_fixed, batch_size=16, shuffle=True, num_workers=0)
    train_guided_loader = DataLoader(train_guided, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
    test_clean_loader = DataLoader(test_clean, batch_size=16, shuffle=False, num_workers=0)
    test_low_loader = DataLoader(test_low, batch_size=16, shuffle=False, num_workers=0)

    results = [
        _run_config(
            "fixed_mobilenet_pretrained",
            "mobilenet_v3_small",
            True,
            MAIN_BENCHMARK_DIR / "fixed" / "best_model.pt",
            train_fixed_loader,
            val_loader,
            test_clean_loader,
            test_low_loader,
            device,
            num_classes,
        ),
        _run_config(
            "guided_mobilenet_pretrained",
            "mobilenet_v3_small",
            True,
            MAIN_BENCHMARK_DIR / "guided" / "best_model.pt",
            train_guided_loader,
            val_loader,
            test_clean_loader,
            test_low_loader,
            device,
            num_classes,
        ),
        _run_config(
            "guided_mobilenet_scratch",
            "mobilenet_v3_small",
            False,
            None,
            train_guided_loader,
            val_loader,
            test_clean_loader,
            test_low_loader,
            device,
            num_classes,
        ),
        _run_config(
            "guided_efficientnet_pretrained",
            "efficientnet_b0",
            True,
            None,
            train_guided_loader,
            val_loader,
            test_clean_loader,
            test_low_loader,
            device,
            num_classes,
        ),
    ]

    payload = {"dataset": "beans", "device": str(device), "results": results}
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
