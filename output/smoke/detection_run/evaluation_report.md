# Prompt2Model Evaluation Report

## Prompt
Detect squares and circles in low light images and prioritize speed.

## Task Summary
- Task: `detection`
- Model: `ssdlite320_mobilenet_v3_large`
- Dataset root: `output/smoke/detection_data`
- Augmentations: `low_light`

## Resolved Labels
- `squares` -> `square` (method=lexical, score=0.923)
- `circles` -> `circle` (method=lexical, score=0.923)

## Metrics
```json
{
  "map@0.5": 0.020833333333333332,
  "map@[0.5:0.95]": 0.0044956140350877185,
  "latency_ms": 66.26199958845973,
  "fps": 15.091606142446707,
  "parameter_count": 2220380.0,
  "parameter_count_millions": 2.22038
}
```

## Training History
```json
[
  {
    "epoch": 1.0,
    "train_loss": 10.901141166687012,
    "val_loss": 8.816788673400879
  }
]
```

## Export
- ONNX artifact: `output/smoke/detection_run/model.onnx`
