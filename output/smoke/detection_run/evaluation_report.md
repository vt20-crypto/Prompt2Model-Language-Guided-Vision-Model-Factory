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
  "map@0.5": 0.07142857142857142,
  "map@[0.5:0.95]": 0.02142857142857143,
  "latency_ms": 21.50512479711324,
  "fps": 46.50054391380402,
  "parameter_count": 2220380.0,
  "parameter_count_millions": 2.22038
}
```

## Training History
```json
[
  {
    "epoch": 1.0,
    "train_loss": 11.06933307647705,
    "val_loss": 9.328075408935547
  }
]
```

## Export
- ONNX artifact: `output/smoke/detection_run/model.onnx`
