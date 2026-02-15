#!/usr/bin/env python3
"""
Script to verify LIAR dataset is correctly loaded
Run this before training to check your data
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader_and_processer import load_liar_dataset, get_tokenizer, preprocess_data
import src.config as config

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                   LIAR Dataset Verification                          ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n📋 Checking dataset files...")
    print("="*70)
    
    # Check files exist
    required_files = {
        'train.tsv': config.RAW_DATA_DIR / 'train.tsv',
        'valid.tsv': config.RAW_DATA_DIR / 'valid.tsv',
        'test.tsv': config.RAW_DATA_DIR / 'test.tsv'
    }
    
    all_exist = True
    for name, path in required_files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✓ {name:12} - Found ({size_mb:.2f} MB)")
        else:
            print(f"✗ {name:12} - MISSING")
            all_exist = False
    
    if not all_exist:
        print("\n" + "="*70)
        print("❌ ERROR: Some dataset files are missing!")
        print("="*70)
        print(f"\nPlease place LIAR dataset files in:")
        print(f"  {config.RAW_DATA_DIR}")
        print("\nRequired files:")
        print("  - train.tsv")
        print("  - valid.tsv")
        print("  - test.tsv")
        print("\nYou can download the LIAR dataset from:")
        print("  https://www.cs.ucsb.edu/~william/data/liar_dataset.zip")
        return False
    
    print("\n" + "="*70)
    print("✓ All dataset files found!")
    print("="*70)
    
    # Try to load data
    print("\n📊 Loading and analyzing dataset...")
    print("="*70)
    
    try:
        train_df, val_df, test_df = load_liar_dataset()
        
        # Show detailed statistics
        print("\n📈 Dataset Statistics:")
        print("="*70)
        
        total = len(train_df) + len(val_df) + len(test_df)
        print(f"\nTotal examples: {total:,}")
        print(f"  Training:   {len(train_df):,} ({len(train_df)/total*100:.1f}%)")
        print(f"  Validation: {len(val_df):,} ({len(val_df)/total*100:.1f}%)")
        print(f"  Test:       {len(test_df):,} ({len(test_df)/total*100:.1f}%)")
        
        # Label distribution
        print("\n🏷️  Label Distribution (Training Set):")
        print("-"*70)
        label_counts = train_df['label'].value_counts().sort_index()
        for label_id, count in label_counts.items():
            label_name = config.LABEL_MAP[label_id]
            percentage = count / len(train_df) * 100
            bar = "█" * int(percentage / 2)
            print(f"{label_id} - {label_name:15} | {count:5,} ({percentage:5.1f}%) {bar}")
        
        # Show sample statements
        print("\n📝 Sample Statements:")
        print("="*70)
        
        for i in range(min(5, len(train_df))):
            label = train_df.iloc[i]['label']
            label_name = config.LABEL_MAP[label]
            statement = train_df.iloc[i]['statement']
            
            # Truncate long statements
            if len(statement) > 100:
                statement = statement[:97] + "..."
            
            print(f"\n{i+1}. [{label_name}]")
            print(f"   {statement}")
        
        # Test tokenization
        print("\n🔤 Testing tokenization...")
        print("="*70)
        
        tokenizer = get_tokenizer()
        sample_text = train_df.iloc[0]['statement']
        tokens = tokenizer.tokenize(sample_text)
        
        print(f"\nSample text: {sample_text[:100]}...")
        print(f"Number of tokens: {len(tokens)}")
        print(f"First 10 tokens: {tokens[:10]}")
        
        # Test preprocessing
        print("\n⚙️  Testing preprocessing pipeline...")
        print("="*70)
        
        # Use small subset for testing
        test_train = train_df.head(100)
        test_val = val_df.head(20)
        test_test = test_df.head(20)
        
        train_dataset, val_dataset, test_dataset = preprocess_data(
            test_train, test_val, test_test, tokenizer
        )
        
        # Get a sample
        sample = train_dataset[0]
        print(f"\nSample dataset item:")
        print(f"  Input IDs shape: {sample['input_ids'].shape}")
        print(f"  Attention mask shape: {sample['attention_mask'].shape}")
        print(f"  Label: {sample['labels'].item()} ({config.LABEL_MAP[sample['labels'].item()]})")
        
        print("\n" + "="*70)
        print("✅ SUCCESS! Dataset is correctly formatted and ready for training")
        print("="*70)
        
        print("\n🚀 Next steps:")
        print("  1. Review the statistics above")
        print("  2. Run training: python train_local.py")
        print("  3. Monitor progress in logs/training_results.json")
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ ERROR: Failed to load dataset")
        print("="*70)
        print(f"\nError details: {e}")
        print("\nPossible issues:")
        print("  - Incorrect file format (must be TSV)")
        print("  - Missing columns in TSV files")
        print("  - Encoding issues")
        print("\nTry opening the files manually to verify format:")
        print(f"  head -n 5 {config.RAW_DATA_DIR}/train.tsv")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
