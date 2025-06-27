#!/usr/bin/env python3
"""
Minimal FastAPI application for Railway deployment testing
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from datetime import datetime

# Create FastAPI app
app = FastAPI(
    title="SomonGPT Real Estate API - Minimal",
    description="Minimal version for deployment testing",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SomonGPT Real Estate API - Minimal Version",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "SomonGPT Real Estate API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/test")
async def test_endpoint():
    """Test endpoint"""
    return {
        "test": "success",
        "environment": {
            "PORT": os.getenv("PORT", "not set"),
            "PYTHON_VERSION": os.getenv("PYTHON_VERSION", "not set"),
            "pwd": os.getcwd()
        }
    }

# Run with uvicorn if called directly
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 