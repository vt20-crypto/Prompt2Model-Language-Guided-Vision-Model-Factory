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
- `blue circle` -> `blue_circle` (method=lexical, score=0.909)
- `green triangle` -> `green_triangle` (method=lexical, score=0.929)

## Metrics
```json
{
  "accuracy": 0.6666666666666666,
  "macro_f1": 0.4,
  "latency_ms": 111.469550000038,
  "fps": 8.971059809604139,
  "parameter_count": 1520931.0,
  "parameter_count_millions": 1.520931
}
```

## Training History
```json
[
  {
    "epoch": 1.0,
    "train_loss": 0.8922191560268402,
    "train_accuracy": 0.6538461538461539,
    "val_loss": 1.1205500364303589,
    "val_accuracy": 0.0
  },
  {
    "epoch": 2.0,
    "train_loss": 0.3471984574571252,
    "train_accuracy": 0.8461538461538461,
    "val_loss": 1.1181145906448364,
    "val_accuracy": 0.0
  }
]
```

## Export
- ONNX artifact: `output/smoke/classification_run/model.onnx`
