#!/usr/bin/env python3
"""
Railway-specific startup script with extensive logging
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_environment():
    """Debug the Railway environment"""
    logger.info("=== RAILWAY STARTUP DEBUG ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")
    
    # Environment variables
    env_vars = ['PORT', 'PYTHONPATH', 'PYTHON_VERSION', 'ENVIRONMENT']
    for var in env_vars:
        logger.info(f"ENV {var}: {os.getenv(var, 'NOT SET')}")
    
    # Check file structure
    current_dir = Path(".")
    logger.info(f"Files in current directory: {list(current_dir.iterdir())}")
    
    # Check if key files exist
    key_files = [
        "integrated_main.py",
        "../../rental_prediction/models/xgboost_model.joblib",
        "../../rental_prediction/data/features/engineered_features.csv"
    ]
    
    for file_path in key_files:
        full_path = Path(file_path)
        exists = full_path.exists()
        logger.info(f"File check {file_path}: {'EXISTS' if exists else 'MISSING'}")
        if exists and full_path.is_file():
            logger.info(f"  Size: {full_path.stat().st_size} bytes")

def test_imports():
    """Test critical imports"""
    logger.info("=== TESTING IMPORTS ===")
    
    imports_to_test = [
        'fastapi',
        'uvicorn',
        'pandas',
        'sqlite3',
        'joblib',
        'sklearn'
    ]
    
    for module_name in imports_to_test:
        try:
            __import__(module_name)
            logger.info(f"✅ {module_name} imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import {module_name}: {e}")

def test_app_import():
    """Test importing the main application"""
    logger.info("=== TESTING APP IMPORT ===")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        from integrated_main import app
        logger.info(f"✅ App imported successfully: {app.title}")
        
        # Test routes
        routes = [route.path for route in app.routes]
        logger.info(f"✅ Found {len(routes)} routes")
        
        return app
    except Exception as e:
        logger.error(f"❌ Failed to import app: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def start_server():
    """Start the uvicorn server"""
    logger.info("=== STARTING SERVER ===")
    
    try:
        import uvicorn
        
        # Get port from environment
        port = int(os.getenv("PORT", 8000))
        logger.info(f"Starting server on port {port}")
        
        # Import the app
        app = test_app_import()
        if app is None:
            logger.error("Cannot start server - app import failed")
            sys.exit(1)
        
        logger.info("🚀 Starting uvicorn server...")
        uvicorn.run(app, host="0.0.0.0", port=port)
        
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    debug_environment()
    test_imports()
    start_server() 