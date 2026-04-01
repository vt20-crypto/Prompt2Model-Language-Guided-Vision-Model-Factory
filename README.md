# Prompt2Model

`Prompt2Model` is a week-4 integrated prototype for the course project "Prompt2Model: Language-Guided Vision Model Factory".

What is implemented:

- Dev Sanghvi track: typed pipeline schema, deterministic prompt parser, augmentation mapping, CLIP-ready label resolver with lexical fallback
- Venkata track: classification and COCO-style detection dataset loaders, lightweight model registry, training loops, augmentation injection
- Madhuvani track: metric harness, ONNX export with metadata injection, ONNX verification, markdown evaluation report generation

What is validated:

- Classification path runs end-to-end on synthetic data: prompt -> config -> training -> metrics -> ONNX export -> report
- Detection path is integrated through the week-4 milestone: prompt -> config -> COCO loader -> detector training/eval smoke test

## Repo Layout

- `src/prompt2model/`: package source
- `tests/`: smoke tests and unit tests
- `latex/`: proposal PDF and LaTeX source
- `docs/`: project package PDF and generated markdown notes

## Environment

The quickest reproducible setup in this repo is a local virtualenv that can still see the machine's installed PyTorch build:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e .
```

## Quick Start

Generate toy datasets:

```bash
.venv/bin/python -m prompt2model.cli generate-toy-data --task all --output-dir output/toy_data
```

Run the built-in smoke demo:

```bash
.venv/bin/python -m prompt2model.cli smoke-test --output-dir output/smoke
```

Run a manual classification job:

```bash
.venv/bin/python -m prompt2model.cli run \
  --prompt "Classify red square, blue circle, and green triangle images under low light and prioritize speed." \
  --task classification \
  --dataset-root output/toy_data/classification \
  --dataset-format imagefolder \
  --output-dir output/manual_classification
```

Run tests:

```bash
.venv/bin/python -m pytest
```

## Notes

- CLIP-based label resolution is implemented behind a lazy loader. It falls back to lexical matching if the CLIP model is unavailable.
- The default augmentation backend is torchvision-native for stability. The module boundary is ready for an Albumentations adapter later.
- Detection ONNX export is present as a best-effort wrapper, but the fully validated export path in this milestone is classification.

