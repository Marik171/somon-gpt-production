#!/usr/bin/env python3
"""
Simple startup test to verify all imports work correctly
"""

import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all critical imports"""
    try:
        logger.info("Testing FastAPI import...")
        from fastapi import FastAPI
        logger.info("✅ FastAPI imported successfully")
        
        logger.info("Testing uvicorn import...")
        import uvicorn
        logger.info("✅ uvicorn imported successfully")
        
        logger.info("Testing pandas import...")
        import pandas as pd
        logger.info("✅ pandas imported successfully")
        
        logger.info("Testing sqlite3 import...")
        import sqlite3
        logger.info("✅ sqlite3 imported successfully")
        
        logger.info("Testing application import...")
        from integrated_main import app
        logger.info("✅ Application imported successfully")
        
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_basic_startup():
    """Test basic application startup"""
    try:
        logger.info("Testing application creation...")
        from integrated_main import app
        logger.info(f"✅ Application created: {app.title}")
        
        # Test that the app has routes
        routes = [route.path for route in app.routes]
        logger.info(f"✅ Found {len(routes)} routes")
        logger.info(f"Sample routes: {routes[:5]}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Application startup failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🧪 Starting startup tests...")
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    # Test basic startup
    if not test_basic_startup():
        sys.exit(1)
    
    logger.info("✅ All startup tests passed!")
    logger.info("Application should be ready for deployment") 