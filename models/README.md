# Models Directory

This directory stores trained models and model artifacts.

## Structure:

```
models/
├── roberta_fact_checker/    # Trained RoBERTa model
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files
├── cache/                    # HuggingFace model cache
└── README.md                 # This file
```

## Usage:

After training, the fine-tuned model will be saved to `roberta_fact_checker/`.

The API service loads the model from this directory.

## Model Files:

- `config.json` - Model configuration
- `pytorch_model.bin` - Model weights
- `tokenizer_config.json` - Tokenizer configuration
- `vocab.json` - Vocabulary
- `merges.txt` - BPE merges

These files are automatically generated during training.
