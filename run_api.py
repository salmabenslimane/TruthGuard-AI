#!/usr/bin/env python3
"""
Run the FastAPI application locally for prediction service
"""
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

import uvicorn
from api.main import app
import src.config as config

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                   TruthGuard-AI Prediction API                       ║
║              Starting FastAPI Service (Local)                         ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"\n🚀 Starting API server...")
    print(f"   Host: {config.API_HOST}")
    print(f"   Port: {config.API_PORT}")
    print(f"   Model: {config.MODEL_PATH}")
    print(f"\n📚 Documentation: http://localhost:{config.API_PORT}/docs")
    print(f"🔍 Health check: http://localhost:{config.API_PORT}/health")
    print("\n" + "="*70 + "\n")
    
    uvicorn.run(
        app, 
        host=config.API_HOST, 
        port=config.API_PORT,
        log_level="info"
    )
