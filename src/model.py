"""
Model definition for TruthGuard-AI
RoBERTa-based fact-checking classifier
"""
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
import config


class FactCheckModel:
    """Wrapper class for the RoBERTa fact-checking model"""
    
    def __init__(self, model_name=config.MODEL_NAME, num_labels=config.NUM_LABELS):
        """
        Initialize the model
        
        Args:
            model_name: Name of the pretrained model
            num_labels: Number of classification labels
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def create_model(self):
        """Create and initialize the model"""
        print(f"Creating model: {self.model_name}")
        print(f"Number of labels: {self.num_labels}")
        print(f"Device: {self.device}")
        print(f"Cache directory: {config.MODEL_CACHE_DIR}")
        
        # Load model configuration
        model_config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="single_label_classification",
            cache_dir=config.MODEL_CACHE_DIR
        )
        
        # Load pretrained model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=model_config,
            cache_dir=config.MODEL_CACHE_DIR
        )
        
        # Move to device
        self.model.to(self.device)
        
        print(f"✓ Model loaded and moved to {self.device}")
        
        return self.model
    
    def load_model(self, model_path):
        """Load a trained model from disk"""
        print(f"Loading model from: {model_path}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=self.num_labels
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded and set to eval mode")
        
        return self.model
    
    def save_model(self, save_path):
        """Save the model to disk"""
        print(f"Saving model to: {save_path}")
        self.model.save_pretrained(save_path)
        print(f"✓ Model saved successfully")
    
    def get_model_info(self):
        """Get information about the model"""
        if self.model is None:
            return "Model not initialized"
        
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "total_parameters": f"{num_params:,}",
            "trainable_parameters": f"{num_trainable:,}",
            "device": str(self.device),
            "cache_dir": str(config.MODEL_CACHE_DIR)
        }
        
        return info


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    print("="*70)
    print("Testing model creation...")
    print("="*70)
    
    fact_model = FactCheckModel()
    model = fact_model.create_model()
    
    info = fact_model.get_model_info()
    print("\n" + "="*70)
    print("Model Information:")
    print("="*70)
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Model creation test PASSED")