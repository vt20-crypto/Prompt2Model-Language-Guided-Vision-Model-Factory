# Prompt2Model Evaluation Report

## Prompt
Classify red square, blue circle, and green triangle images under low light and prioritize speed.

## Task Summary
- Task: `classification`
- Model: `mobilenet_v3_small`
- Dataset root: `output/smoke/classification_data`
- Augmentations: `low_light`

## Resolved Labels
- `red square` -> `red_square` (method=lexical, score=0.900)

## Metrics
```json
{
  "accuracy": 0.3333333333333333,
  "macro_f1": 0.25,
  "latency_ms": 457.0540581946261,
  "fps": 2.1879249993972762,
  "parameter_count": 1520931.0,
  "parameter_count_millions": 1.520931
}
```

## Training History
```json
[
  {
    "epoch": 1.0,
    "train_loss": 1.159813016653061,
    "train_accuracy": 0.6153846153846154,
    "val_loss": 1.0960280895233154,
    "val_accuracy": 0.2857142857142857
  },
  {
    "epoch": 2.0,
    "train_loss": 0.5010360181331635,
    "train_accuracy": 0.8461538461538461,
    "val_loss": 1.0888588428497314,
    "val_accuracy": 0.2857142857142857
  }
]
```

## Export
- ONNX artifact: `output/smoke/classification_run/model.onnx`
