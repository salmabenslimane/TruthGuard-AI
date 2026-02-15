"""
Evaluation utilities for TruthGuard-AI
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import config


def compute_metrics(predictions, labels):
    """
    Compute evaluation metrics
    
    Args:
        predictions: Model predictions
        labels: True labels
        
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(labels, predictions)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to use
        
    Returns:
        Predictions, labels, and metrics
    """
    import torch
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    logits = np.array(all_logits)
    
    metrics = compute_metrics(predictions, labels)
    
    return predictions, labels, logits, metrics


def print_classification_report(predictions, labels):
    """Print detailed classification report"""
    target_names = [config.LABEL_MAP[i] for i in range(config.NUM_LABELS)]
    
    print("\nDetailed Classification Report:")
    print("="*70)
    print(classification_report(labels, predictions, target_names=target_names))


def print_confusion_matrix(predictions, labels):
    """Print confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    print("\nConfusion Matrix:")
    print("="*70)
    
    # Print header
    print(f"{'':20}", end="")
    for i in range(config.NUM_LABELS):
        print(f"{config.LABEL_MAP[i]:15}", end="")
    print()
    print("-"*70)
    
    # Print rows
    for i in range(config.NUM_LABELS):
        print(f"{config.LABEL_MAP[i]:20}", end="")
        for j in range(config.NUM_LABELS):
            print(f"{cm[i][j]:15}", end="")
        print()


if __name__ == "__main__":
    # Test metrics computation
    test_preds = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
    test_labels = np.array([0, 1, 2, 3, 4, 4, 0, 1, 1, 3])
    
    metrics = compute_metrics(test_preds, test_labels)
    print("Test Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")