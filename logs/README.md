# Logs Directory

This directory contains training logs and experiment results.

## Files:

- `training_results.json` - Complete training metrics and configuration
- Training logs will be saved here automatically

## Example training_results.json:

```json
{
  "model_name": "roberta-base",
  "num_epochs": 3,
  "batch_size": 16,
  "learning_rate": 2e-5,
  "training_time": 1234.56,
  "best_val_accuracy": 0.8765,
  "test_metrics": {
    "accuracy": 0.8543,
    "f1_weighted": 0.8512,
    "f1_macro": 0.8234
  },
  "training_history": {
    "train_loss": [0.45, 0.32, 0.21],
    "train_accuracy": [0.78, 0.84, 0.89],
    "val_loss": [0.42, 0.35, 0.28],
    "val_accuracy": [0.81, 0.86, 0.88],
    "val_f1": [0.80, 0.85, 0.87]
  },
  "timestamp": "2024-01-15T10:30:00"
}
```
