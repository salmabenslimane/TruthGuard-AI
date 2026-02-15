"""
Test client for TruthGuard-AI API
"""
import requests
import json

API_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_labels():
    """Test labels endpoint"""
    print("Testing labels endpoint...")
    response = requests.get(f"{API_URL}/labels")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_predict(statement):
    """Test prediction endpoint"""
    print(f"Testing prediction for: '{statement}'")
    
    data = {
        "statement": statement,
        "return_probabilities": True
    }
    
    response = requests.post(f"{API_URL}/predict", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_batch_predict(statements):
    """Test batch prediction endpoint"""
    print(f"Testing batch prediction for {len(statements)} statements...")
    
    data = {
        "statements": statements,
        "return_probabilities": False
    }
    
    response = requests.post(f"{API_URL}/batch_predict", json=data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"\nResults:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. Statement: {result['statement']}")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.4f}")
    else:
        print(f"Error: {response.text}")
    print()


if __name__ == "__main__":
    print("="*70)
    print("TruthGuard-AI API Test Client")
    print("="*70)
    print()
    
    # Test health
    test_health()
    
    # Test labels
    test_labels()
    
    # Test single predictions
    test_statements = [
        "The Earth orbits around the Sun.",
        "Vaccines cause autism in children.",
        "Climate change is primarily caused by human activities.",
        "The Earth is flat.",
        "Water freezes at 0 degrees Celsius."
    ]
    
    for statement in test_statements:
        test_predict(statement)
    
    # Test batch prediction
    print("="*70)
    print("Batch Prediction Test")
    print("="*70)
    print()
    
    test_batch_predict(test_statements)
