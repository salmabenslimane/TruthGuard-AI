"""
Data loading and preprocessing for TruthGuard-AI
Loads the LIAR dataset from local TSV files and saves processed data
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pathlib import Path
import config

class FactCheckDataset(Dataset):
    """Custom Dataset for fact-checking"""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


def load_liar_tsv(file_path):
    """
    Load and preprocess LIAR dataset from TSV file
    
    LIAR dataset format (tab-separated):
    Column 0: ID
    Column 1: label (pants-fire, false, barely-true, half-true, mostly-true, true)
    Column 2: statement (the text we want to classify)
    Columns 3-13: metadata (speaker, context, etc.) - WE IGNORE THESE
    
    Preprocessing steps:
    1. Load all 14 columns from TSV
    2. Keep ONLY 'statement' and 'label' columns
    3. Convert string labels to integers (0-5)
    4. Remove rows with invalid/missing data
    5. Clean text (strip whitespace)
    
    Args:
        file_path: Path to the TSV file
        
    Returns:
        DataFrame with ONLY 'statement' and 'label' columns (cleaned)
    """
    print(f"Loading data from: {file_path}")
    
    # Column names for LIAR dataset (14 columns total)
    column_names = [
        'id', 'label', 'statement', 'subject', 'speaker', 'speaker_job',
        'state_info', 'party_affiliation', 'barely_true_counts',
        'false_counts', 'half_true_counts', 'mostly_true_counts',
        'pants_on_fire_counts', 'context'
    ]
    
    # Step 1: Read TSV file with all columns
    df = pd.read_csv(file_path, sep='\t', header=None, names=column_names)
    initial_size = len(df)
    print(f"  Loaded {initial_size} rows (raw)")
    
    # Step 2: PREPROCESSING - Keep only what we need
    print(f"  Preprocessing: Keeping only 'statement' and 'label' columns")
    df = df[['statement', 'label']].copy()
    
    # Step 3: Clean text data
    print(f"  Cleaning: Stripping whitespace from statements")
    df['statement'] = df['statement'].str.strip()
    
    # Step 4: Remove empty statements
    df = df[df['statement'].str.len() > 0]
    removed_empty = initial_size - len(df)
    if removed_empty > 0:
        print(f"  Removed {removed_empty} empty statements")
    
    # Step 5: Map string labels to integers
    print(f"  Converting labels: string → integer (0-5)")
    df['label'] = df['label'].map(config.LIAR_LABEL_MAPPING)
    
    # Step 6: Remove rows with invalid labels
    before_drop = len(df)
    df = df.dropna(subset=['label'])
    invalid_labels = before_drop - len(df)
    if invalid_labels > 0:
        print(f"  Removed {invalid_labels} rows with invalid labels")
    
    df['label'] = df['label'].astype(int)
    
    # Final statistics
    print(f"  ✓ Final dataset: {len(df)} examples ({initial_size - len(df)} removed)")
    print(f"\n  Label distribution:")
    label_counts = df['label'].value_counts().sort_index()
    for label_id, count in label_counts.items():
        label_name = config.LABEL_MAP[label_id]
        print(f"    {label_id} ({label_name:15}): {count:5,} ({count/len(df)*100:5.1f}%)")
    
    return df


def save_processed_data(train_df, val_df, test_df):
    """
    Save preprocessed data to CSV files in processed folder
    
    Args:
        train_df, val_df, test_df: Preprocessed DataFrames
    """
    print("\n" + "="*70)
    print("Saving preprocessed data...")
    print("="*70)
    
    # Save to CSV files
    print(f"Saving training data to: {config.PROCESSED_TRAIN_FILE}")
    train_df.to_csv(config.PROCESSED_TRAIN_FILE, index=False)
    
    print(f"Saving validation data to: {config.PROCESSED_VAL_FILE}")
    val_df.to_csv(config.PROCESSED_VAL_FILE, index=False)
    
    print(f"Saving test data to: {config.PROCESSED_TEST_FILE}")
    test_df.to_csv(config.PROCESSED_TEST_FILE, index=False)
    
    print(f"\n✓ Preprocessed data saved successfully!")
    print(f"  Total files: 3")
    print(f"  Location: {config.PROCESSED_DATA_DIR}")


def load_processed_data():
    """
    Load preprocessed data from CSV files if they exist
    
    Returns:
        train_df, val_df, test_df if files exist, otherwise None
    """
    # Check if all processed files exist
    if (config.PROCESSED_TRAIN_FILE.exists() and 
        config.PROCESSED_VAL_FILE.exists() and 
        config.PROCESSED_TEST_FILE.exists()):
        
        print("="*70)
        print("Loading preprocessed data from saved files...")
        print("="*70)
        
        train_df = pd.read_csv(config.PROCESSED_TRAIN_FILE)
        val_df = pd.read_csv(config.PROCESSED_VAL_FILE)
        test_df = pd.read_csv(config.PROCESSED_TEST_FILE)
        
        print(f"✓ Loaded preprocessed data:")
        print(f"  Train: {len(train_df):,} examples")
        print(f"  Valid: {len(val_df):,} examples")
        print(f"  Test:  {len(test_df):,} examples")
        
        return train_df, val_df, test_df
    
    return None


def load_liar_dataset(use_cached=True):
    """
    Load the LIAR dataset from local TSV files or preprocessed cache
    
    Args:
        use_cached: If True, load from processed cache if available
    
    Returns:
        train_df, val_df, test_df: DataFrames with 'statement' and 'label' columns
    """
    # Try to load preprocessed data first
    if use_cached:
        processed_data = load_processed_data()
        if processed_data is not None:
            print("\n💾 Using cached preprocessed data (fast loading)")
            print("   To reprocess from raw TSV: set use_cached=False")
            return processed_data
    
    # Load from raw TSV files
    print("="*70)
    print("Loading LIAR dataset from RAW TSV files")
    print("="*70)
    
    # Define file paths
    train_path = config.RAW_DATA_DIR / "train.tsv"
    val_path = config.RAW_DATA_DIR / "valid.tsv"
    test_path = config.RAW_DATA_DIR / "test.tsv"
    
    # Check if files exist
    missing_files = []
    for file_path in [train_path, val_path, test_path]:
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        error_msg = f"Missing LIAR dataset files:\n" + "\n".join(f"  - {f}" for f in missing_files)
        error_msg += f"\n\nPlease place the LIAR dataset files in: {config.RAW_DATA_DIR}"
        error_msg += "\nExpected files: train.tsv, valid.tsv, test.tsv"
        raise FileNotFoundError(error_msg)
    
    # Load and preprocess datasets
    print("\n1. Loading training data...")
    train_df = load_liar_tsv(train_path)
    
    print("\n2. Loading validation data...")
    val_df = load_liar_tsv(val_path)
    
    print("\n3. Loading test data...")
    test_df = load_liar_tsv(test_path)
    
    # Summary
    print("\n" + "="*70)
    print("Dataset Summary:")
    print("="*70)
    print(f"Train size: {len(train_df):,}")
    print(f"Validation size: {len(val_df):,}")
    print(f"Test size: {len(test_df):,}")
    print(f"Total: {len(train_df) + len(val_df) + len(test_df):,}")
    
    # Save preprocessed data for future use
    save_processed_data(train_df, val_df, test_df)
    
    return train_df, val_df, test_df


def preprocess_data(train_df, val_df, test_df, tokenizer):
    """
    Preprocess and tokenize the datasets
    
    Args:
        train_df, val_df, test_df: DataFrames with 'statement' and 'label'
        tokenizer: HuggingFace tokenizer
        
    Returns:
        train_dataset, val_dataset, test_dataset: PyTorch datasets
    """
    print("\n" + "="*70)
    print("Tokenizing data for model...")
    print("="*70)
    
    # Extract texts and labels
    train_texts = train_df['statement'].tolist()
    train_labels = train_df['label'].tolist()
    
    val_texts = val_df['statement'].tolist()
    val_labels = val_df['label'].tolist()
    
    test_texts = test_df['statement'].tolist()
    test_labels = test_df['label'].tolist()
    
    # Tokenize
    print("\n1. Tokenizing training data...")
    train_encodings = tokenizer(
        train_texts, 
        truncation=True, 
        padding=True, 
        max_length=config.MAX_LENGTH
    )
    
    print("2. Tokenizing validation data...")
    val_encodings = tokenizer(
        val_texts, 
        truncation=True, 
        padding=True,
        max_length=config.MAX_LENGTH
    )
    
    print("3. Tokenizing test data...")
    test_encodings = tokenizer(
        test_texts, 
        truncation=True, 
        padding=True,
        max_length=config.MAX_LENGTH
    )
    
    # Create datasets
    print("\n4. Creating PyTorch datasets...")
    train_dataset = FactCheckDataset(train_encodings, train_labels)
    val_dataset = FactCheckDataset(val_encodings, val_labels)
    test_dataset = FactCheckDataset(test_encodings, test_labels)
    
    print(f"\n✓ Tokenization complete!")
    print(f"  Train dataset: {len(train_dataset):,} examples")
    print(f"  Val dataset: {len(val_dataset):,} examples")
    print(f"  Test dataset: {len(test_dataset):,} examples")
    
    return train_dataset, val_dataset, test_dataset


def get_tokenizer():
    """Load the tokenizer for the model"""
    print(f"\nLoading tokenizer: {config.MODEL_NAME}")
    print(f"Cache directory: {config.MODEL_CACHE_DIR}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        cache_dir=config.MODEL_CACHE_DIR
    )
    
    print(f"✓ Tokenizer loaded")
    return tokenizer


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading pipeline...")
    print("="*70)
    
    try:
        # Load datasets (will use cache if available)
        train_df, val_df, test_df = load_liar_dataset()
        
        # Show sample data
        print("\n" + "="*70)
        print("Sample from training data:")
        print("="*70)
        for i in range(min(3, len(train_df))):
            label = train_df.iloc[i]['label']
            label_name = config.LABEL_MAP[label]
            statement = train_df.iloc[i]['statement']
            if len(statement) > 100:
                statement = statement[:97] + "..."
            print(f"\n{i+1}. Label: {label_name} ({label})")
            print(f"   Statement: {statement}")
        
        # Load tokenizer
        tokenizer = get_tokenizer()
        
        # Preprocess data
        train_dataset, val_dataset, test_dataset = preprocess_data(
            train_df, val_df, test_df, tokenizer
        )
        
        print("\n" + "="*70)
        print("✓ Data loading pipeline test PASSED")
        print("="*70)
        
    except FileNotFoundError as e:
        print("\n" + "="*70)
        print("✗ Data loading pipeline test FAILED")
        print("="*70)
        print(f"\nError: {e}")
        print("\nPlease ensure LIAR dataset files are placed in:")
        print(f"  {config.RAW_DATA_DIR}")
        print("\nRequired files:")
        print("  - train.tsv")
        print("  - valid.tsv")
        print("  - test.tsv")