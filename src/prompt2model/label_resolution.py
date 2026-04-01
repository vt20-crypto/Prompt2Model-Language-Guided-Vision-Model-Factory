from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import torch

from prompt2model.config import RequestedLabel, ResolvedLabel


@dataclass
class _ClipCache:
    tokenizer: Any
    model: Any


class LabelResolver:
    def __init__(
        self,
        threshold: float = 0.25,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        enable_clip: bool = True,
    ) -> None:
        self.threshold = threshold
        self.clip_model_name = clip_model_name
        self.enable_clip = enable_clip
        self._cache: _ClipCache | None = None

    def resolve(
        self,
        requested_labels: list[RequestedLabel],
        dataset_labels: list[str],
    ) -> list[ResolvedLabel]:
        if not dataset_labels:
            return [
                ResolvedLabel(
                    requested_label=request.name,
                    dataset_label=request.name,
                    score=1.0,
                    method="identity",
                )
                for request in requested_labels
            ]

        if self.enable_clip:
            try:
                return self._resolve_semantic(requested_labels, dataset_labels)
            except Exception:
                pass
        return self._resolve_lexical(requested_labels, dataset_labels)

    def _resolve_lexical(
        self,
        requested_labels: list[RequestedLabel],
        dataset_labels: list[str],
    ) -> list[ResolvedLabel]:
        resolved: list[ResolvedLabel] = []
        for request in requested_labels:
            best_label = ""
            best_score = -1.0
            phrases = [request.name, *request.synonyms]
            for phrase in phrases:
                for dataset_label in dataset_labels:
                    score = SequenceMatcher(None, phrase.lower(), dataset_label.lower()).ratio()
                    if score > best_score:
                        best_score = score
                        best_label = dataset_label
            resolved.append(
                ResolvedLabel(
                    requested_label=request.name,
                    dataset_label=best_label or dataset_labels[0],
                    score=float(best_score),
                    method="lexical",
                )
            )
        return resolved

    def _load_clip(self) -> _ClipCache:
        if self._cache is not None:
            return self._cache
        from transformers import AutoTokenizer, CLIPModel

        tokenizer = AutoTokenizer.from_pretrained(self.clip_model_name)
        model = CLIPModel.from_pretrained(self.clip_model_name)
        model.eval()
        self._cache = _ClipCache(tokenizer=tokenizer, model=model)
        return self._cache

    def _embed_texts(self, texts: list[str]) -> torch.Tensor:
        cache = self._load_clip()
        inputs = cache.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            features = cache.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(features, dim=-1)

    def _resolve_semantic(
        self,
        requested_labels: list[RequestedLabel],
        dataset_labels: list[str],
    ) -> list[ResolvedLabel]:
        dataset_embeddings = self._embed_texts(dataset_labels)
        resolved: list[ResolvedLabel] = []

        for request in requested_labels:
            phrases = [request.name, *request.synonyms]
            request_embedding = self._embed_texts(phrases).mean(dim=0, keepdim=True)
            request_embedding = torch.nn.functional.normalize(request_embedding, dim=-1)
            similarities = (request_embedding @ dataset_embeddings.T).squeeze(0)
            best_index = int(torch.argmax(similarities).item())
            best_score = float(similarities[best_index].item())
            if best_score < self.threshold:
                lexical = self._resolve_lexical([request], dataset_labels)[0]
                resolved.append(lexical)
                continue
            resolved.append(
                ResolvedLabel(
                    requested_label=request.name,
                    dataset_label=dataset_labels[best_index],
                    score=best_score,
                    method="clip",
                )
            )

        return resolved

