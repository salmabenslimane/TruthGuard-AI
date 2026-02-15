"""
Training script for TruthGuard-AI
Fine-tunes RoBERTa on the fact-checking task
"""
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import random
import json
import time
from datetime import datetime
import config
from data_loader_and_processer import load_liar_dataset, preprocess_data, get_tokenizer
from model import FactCheckModel, count_parameters
from evaluate import evaluate_model, compute_metrics


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, train_loader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct_predictions/total_predictions:.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions
    
    return avg_loss, accuracy


def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
    
    return avg_loss, metrics


def train_model():
    """Main training function"""
    print("="*50)
    print("TruthGuard-AI Training Pipeline")
    print("="*50)
    
    # Set random seed
    set_seed(config.RANDOM_SEED)
    print(f"\nRandom seed set to: {config.RANDOM_SEED}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data (will use cached processed data if available)
    print("\n" + "="*50)
    print("Loading and preprocessing data...")
    print("="*50)
    train_df, val_df, test_df = load_liar_dataset(use_cached=True)
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    
    # Preprocess data (tokenization)
    train_dataset, val_dataset, test_dataset = preprocess_data(
        train_df, val_df, test_df, tokenizer
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"\nTraining batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\n" + "="*50)
    print("Initializing model...")
    print("="*50)
    fact_model = FactCheckModel()
    model = fact_model.create_model()
    
    model_info = fact_model.get_model_info()
    print("\nModel Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    print(f"\nOptimizer: AdamW")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {config.WARMUP_STEPS}")
    
    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    best_val_accuracy = 0
    training_history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }
    
    start_time = time.time()
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val F1 (weighted): {val_metrics['f1_weighted']:.4f}")
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['train_accuracy'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_accuracy'].append(val_metrics['accuracy'])
        training_history['val_f1'].append(val_metrics['f1_weighted'])
        
        # Save best model
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            save_path = config.MODEL_DIR / "roberta_fact_checker"
            fact_model.save_model(str(save_path))
            tokenizer.save_pretrained(str(save_path))
            print(f"✓ New best model saved! (Accuracy: {best_val_accuracy:.4f})")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("Evaluating on test set...")
    print("="*50)
    
    test_loss, test_metrics = validate(model, test_loader, device)
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
    
    # Save training results
    results = {
        'model_name': config.MODEL_NAME,
        'num_epochs': config.NUM_EPOCHS,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'training_time': training_time,
        'best_val_accuracy': best_val_accuracy,
        'test_metrics': test_metrics,
        'training_history': training_history,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = config.LOG_DIR / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print("\n" + "="*50)
    print("Training pipeline completed successfully!")
    print("="*50)


if __name__ == "__main__":
    train_model()