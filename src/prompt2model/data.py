from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from prompt2model.augmentations import TorchVisionAugmentationBackend
from prompt2model.config import DatasetConfig, DatasetFormat

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _base_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _split_items(items: list[Any], val_split: float, test_split: float, seed: int) -> tuple[list[Any], list[Any], list[Any]]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    test_count = int(math.floor(total * test_split))
    val_count = int(math.floor(total * val_split))
    test_items = shuffled[:test_count]
    val_items = shuffled[test_count : test_count + val_count]
    train_items = shuffled[test_count + val_count :]
    return train_items, val_items, test_items


class ClassificationDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int]],
        class_names: list[str],
        image_size: int,
        augmentations: TorchVisionAugmentationBackend | None = None,
    ) -> None:
        self.samples = samples
        self.class_names = class_names
        self.augmentations = augmentations
        self.transform = _base_transform(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.augmentations is not None:
            image, _ = self.augmentations(image, None)
        image_tensor = self.transform(image)
        return image_tensor, torch.tensor(label, dtype=torch.long)


class CSVClassificationSource:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root

    def load(self) -> tuple[list[tuple[str, int]], list[str]]:
        annotation_path = self.dataset_root / "labels.csv"
        rows = list(csv.DictReader(annotation_path.read_text().splitlines()))
        class_names = sorted({row["label"] for row in rows})
        class_to_index = {label: idx for idx, label in enumerate(class_names)}
        samples = [(str(self.dataset_root / row["image"]), class_to_index[row["label"]]) for row in rows]
        return samples, class_names


class CocoDetectionDataset(Dataset):
    def __init__(
        self,
        root: Path,
        annotation_path: Path,
        image_ids: list[int],
        image_size: int,
        augmentations: TorchVisionAugmentationBackend | None = None,
    ) -> None:
        data = json.loads(annotation_path.read_text())
        self.root = root
        self.image_size = image_size
        self.augmentations = augmentations
        self.images = {item["id"]: item for item in data["images"] if item["id"] in set(image_ids)}
        self.image_ids = image_ids
        self.categories = {item["id"]: item["name"] for item in data["categories"]}
        self.annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in data["annotations"]:
            if annotation["image_id"] in self.images:
                self.annotations_by_image[annotation["image_id"]].append(annotation)
        self.transform = _base_transform(image_size)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image_id = self.image_ids[index]
        image_meta = self.images[image_id]
        image_path = self.root / image_meta["file_name"]
        image = Image.open(image_path).convert("RGB")
        annotations = self.annotations_by_image.get(image_id, [])

        boxes = []
        labels = []
        for annotation in annotations:
            x, y, width, height = annotation["bbox"]
            boxes.append([x, y, x + width, y + height])
            labels.append(annotation["category_id"])

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target: dict[str, Any] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        if boxes:
            target["area"] = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        else:
            target["area"] = torch.zeros((0,), dtype=torch.float32)

        original_width, original_height = image.size
        if self.augmentations is not None:
            image, target = self.augmentations(image, target)

        scale_x = self.image_size / original_width
        scale_y = self.image_size / original_height
        if len(target["boxes"]) > 0:
            boxes_tensor = target["boxes"].clone()
            boxes_tensor[:, [0, 2]] *= scale_x
            boxes_tensor[:, [1, 3]] *= scale_y
            target["boxes"] = boxes_tensor

        image_tensor = self.transform(image)
        return image_tensor, target


@dataclass
class ClassificationBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]


@dataclass
class DetectionBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]


def load_dataset_labels(config: DatasetConfig) -> list[str]:
    root = Path(config.root)
    if config.format == DatasetFormat.IMAGEFOLDER:
        folder = datasets.ImageFolder(root)
        return list(folder.classes)
    if config.format == DatasetFormat.CSV:
        _, class_names = CSVClassificationSource(root).load()
        return class_names
    data = json.loads(Path(config.annotation_path or "").read_text())
    return [item["name"] for item in sorted(data["categories"], key=lambda item: item["id"])]


def build_classification_bundle(
    config: DatasetConfig,
    batch_size: int,
    augmentations: TorchVisionAugmentationBackend | None,
    num_workers: int = 0,
) -> ClassificationBundle:
    root = Path(config.root)
    if config.format == DatasetFormat.IMAGEFOLDER:
        source = datasets.ImageFolder(root)
        samples = [(path, label) for path, label in source.samples]
        class_names = list(source.classes)
    elif config.format == DatasetFormat.CSV:
        samples, class_names = CSVClassificationSource(root).load()
    else:
        raise ValueError("classification bundle requires imagefolder or csv format")

    train_samples, val_samples, test_samples = _split_items(samples, config.val_split, config.test_split, config.seed)
    train_dataset = ClassificationDataset(train_samples, class_names, config.image_size, augmentations=augmentations)
    val_dataset = ClassificationDataset(val_samples, class_names, config.image_size, augmentations=None)
    test_dataset = ClassificationDataset(test_samples, class_names, config.image_size, augmentations=None)
    return ClassificationBundle(
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        class_names=class_names,
    )


def detection_collate_fn(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_detection_bundle(
    config: DatasetConfig,
    batch_size: int,
    augmentations: TorchVisionAugmentationBackend | None,
    num_workers: int = 0,
) -> DetectionBundle:
    annotation_path = Path(config.annotation_path or "")
    data = json.loads(annotation_path.read_text())
    image_ids = [item["id"] for item in data["images"]]
    train_ids, val_ids, test_ids = _split_items(image_ids, config.val_split, config.test_split, config.seed)
    class_names = [item["name"] for item in sorted(data["categories"], key=lambda item: item["id"])]
    root = Path(config.root)

    train_dataset = CocoDetectionDataset(root, annotation_path, train_ids, config.image_size, augmentations=augmentations)
    val_dataset = CocoDetectionDataset(root, annotation_path, val_ids, config.image_size, augmentations=None)
    test_dataset = CocoDetectionDataset(root, annotation_path, test_ids, config.image_size, augmentations=None)
    return DetectionBundle(
        train_loader=DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=detection_collate_fn,
        ),
        val_loader=DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=detection_collate_fn,
        ),
        test_loader=DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=detection_collate_fn,
        ),
        class_names=class_names,
    )


def create_synthetic_classification_dataset(output_dir: str, samples_per_class: int = 12, image_size: int = 96) -> Path:
    root = Path(output_dir)
    classes = ["red_square", "blue_circle", "green_triangle"]
    shape_drawers = {
        "red_square": lambda draw: draw.rectangle([20, 20, image_size - 20, image_size - 20], fill=(220, 40, 40)),
        "blue_circle": lambda draw: draw.ellipse([18, 18, image_size - 18, image_size - 18], fill=(40, 60, 220)),
        "green_triangle": lambda draw: draw.polygon(
            [(image_size // 2, 12), (12, image_size - 12), (image_size - 12, image_size - 12)],
            fill=(60, 180, 80),
        ),
    }

    for class_name in classes:
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(samples_per_class):
            image = Image.new("RGB", (image_size, image_size), color=(245, 245, 245))
            draw = ImageDraw.Draw(image)
            shape_drawers[class_name](draw)
            image.save(class_dir / f"{class_name}_{index:03d}.png")
    return root


def create_synthetic_detection_dataset(output_dir: str, num_images: int = 12, image_size: int = 128) -> tuple[Path, Path]:
    root = Path(output_dir)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    categories = [
        {"id": 1, "name": "square"},
        {"id": 2, "name": "circle"},
    ]
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    rng = random.Random(42)

    for image_id in range(1, num_images + 1):
        file_name = f"image_{image_id:03d}.png"
        image = Image.new("RGB", (image_size, image_size), color=(248, 248, 248))
        draw = ImageDraw.Draw(image)

        square_x = rng.randint(10, 40)
        square_y = rng.randint(10, 40)
        square_size = rng.randint(28, 42)
        draw.rectangle([square_x, square_y, square_x + square_size, square_y + square_size], fill=(225, 50, 50))
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [square_x, square_y, square_size, square_size],
                "area": square_size * square_size,
                "iscrowd": 0,
            }
        )
        annotation_id += 1

        circle_x = rng.randint(58, 82)
        circle_y = rng.randint(58, 82)
        circle_size = rng.randint(24, 36)
        draw.ellipse([circle_x, circle_y, circle_x + circle_size, circle_y + circle_size], fill=(50, 80, 225))
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 2,
                "bbox": [circle_x, circle_y, circle_size, circle_size],
                "area": circle_size * circle_size,
                "iscrowd": 0,
            }
        )
        annotation_id += 1

        image.save(images_dir / file_name)
        images.append({"id": image_id, "file_name": f"images/{file_name}", "width": image_size, "height": image_size})

    annotation_path = root / "annotations.json"
    annotation_path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}, indent=2))
    return images_dir.parent, annotation_path

