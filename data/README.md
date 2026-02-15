# Data Directory

This directory contains the datasets used for training and evaluation.

## Structure:

```
data/
├── raw/          # Original datasets
├── processed/    # Preprocessed data
└── README.md     # This file
```

## Dataset Information:

The project uses the LIAR dataset for fact-checking, which is automatically downloaded from HuggingFace.

If the dataset fails to download, a synthetic dataset will be generated automatically for demonstration purposes.

## Manual Dataset Placement:

If you have your own dataset, place it here and modify `src/data_loader.py` accordingly.
