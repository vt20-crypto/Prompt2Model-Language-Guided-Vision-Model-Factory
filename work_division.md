# Prompt2Model Work Division

## Ownership Domains

- Dev Sanghvi: language/configuration core, prompt parsing, augmentation mapping, label resolution
- Venkata Sai Aneesh Thatiparti: dataset handling, model integration, training loops, augmentation injection
- Madhuvani Thatiparti: evaluation harness, ONNX export, metadata packaging, reporting and reproducibility

## Progress Through Week 4

### Dev Sanghvi

- Week 1 completed: typed pipeline schema in `src/prompt2model/config.py`
- Week 2 completed: prompt-to-config parser in `src/prompt2model/parsing.py`
- Week 3 completed: environment-to-augmentation mapping in `src/prompt2model/augmentations.py`
- Week 4 completed: CLIP-ready label resolver with lexical fallback in `src/prompt2model/label_resolution.py`

### Venkata Sai Aneesh Thatiparti

- Week 1 completed: image-folder, CSV-ready, and COCO-style dataset ingestion in `src/prompt2model/data.py`
- Week 2 completed: lightweight backbone registry in `src/prompt2model/models.py`
- Week 3 completed: baseline classification and detection training loops in `src/prompt2model/training.py`
- Week 4 completed: augmentation injection wired through dataset construction and pipeline orchestration

### Madhuvani Thatiparti

- Week 1 completed: repo packaging and test layout via `pyproject.toml`, `requirements.txt`, `tests/`, and CLI scaffolding
- Week 2 completed: classification metrics and lightweight detection mAP in `src/prompt2model/evaluation.py`
- Week 3 completed: ONNX export and metadata injection in `src/prompt2model/exporting.py`
- Week 4 completed: report generation and integrated pipeline outputs in `src/prompt2model/reporting.py` and `src/prompt2model/pipeline.py`

## Shared Validation

- Unit and smoke tests pass: `6 passed`
- Classification smoke path is validated end-to-end
- Detection smoke path is validated through training and metric computation
- Repo-local smoke artifacts are available under `output/smoke/`

