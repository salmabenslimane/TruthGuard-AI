import pandas as pd
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw')

columns = [
    "id", "label", "statement", "subject", "speaker", 
    "job_title", "state_info", "party_affiliation", 
    "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context"
]

train_df = pd.read_csv(os.path.join(RAW_DIR, 'train.tsv'), sep='\t', names=columns)
val_df   = pd.read_csv(os.path.join(RAW_DIR, 'valid.tsv'), sep='\t', names=columns)
test_df  = pd.read_csv(os.path.join(RAW_DIR, 'test.tsv'), sep='\t', names=columns)

print("Train shape:", train_df.shape)
print(train_df.head())
print("\nValidation shape:", val_df.shape)
print(val_df.head()) 
print("\nTest shape:", test_df.shape) 
print(test_df.head())