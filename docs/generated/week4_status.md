# Prompt2Model Week-4 Status

This repo now covers the integrated scope through week 4 for all three ownership lanes in the project package.

## Dev Sanghvi

- Week 1: `src/prompt2model/config.py` defines the typed configuration schema.
- Week 2: `src/prompt2model/parsing.py` converts prompts into pipeline configs.
- Week 3: `src/prompt2model/augmentations.py` maps environment descriptors into executable augmentation plans.
- Week 4: `src/prompt2model/label_resolution.py` implements a CLIP-ready label resolver with deterministic fallback.

## Venkata Sai Aneesh Thatiparti

- Week 1: `src/prompt2model/data.py` supports image-folder classification and COCO-style detection loading.
- Week 2: `src/prompt2model/models.py` registers lightweight classification and detection backbones.
- Week 3: `src/prompt2model/training.py` implements baseline train/validation loops.
- Week 4: augmentation plans are injected into the dataset layer for both task families.

## Madhuvani Thatiparti

- Week 1: repo packaging, CLI entrypoint, and test layout are in place.
- Week 2: `src/prompt2model/evaluation.py` computes classification metrics and lightweight detection mAP.
- Week 3: `src/prompt2model/exporting.py` exports models to ONNX and injects metadata.
- Week 4: ONNX verification and markdown report generation are wired into `src/prompt2model/pipeline.py`.

## Integrated Result

The classification vertical slice is runnable end-to-end and produces:

- trained checkpoint
- evaluation metrics
- ONNX artifact with embedded metadata
- markdown evaluation report

The detection path is wired through prompt parsing, label resolution, data loading, model instantiation, training, and metric evaluation.

