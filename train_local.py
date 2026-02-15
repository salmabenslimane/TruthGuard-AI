#!/usr/bin/env python3
"""
Local training script for TruthGuard-AI
Run this script to train the model without Docker
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now run the training
from src.train import train_model

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                   TruthGuard-AI Local Training                       ║
║              Training RoBERTa on LIAR Dataset (Local)                ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    train_model()
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                   Training Complete!                                  ║
║                                                                       ║
║  Next steps:                                                          ║
║  1. Check logs/training_results.json for metrics                     ║
║  2. Find trained model in models/roberta_fact_checker/               ║
║  3. Run API: python run_api.py                                       ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
