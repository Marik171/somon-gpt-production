#!/usr/bin/env python3
"""
Full Integration: Enhanced Real Estate Platform

Combines authentication, user management, property analytics, and ML predictions
with the existing real estate data and bargain detection system.
"""

from fastapi import FastAPI, HTTPException, Depends, status, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
import sqlite3
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import asyncio
import uvicorn
import re
import hashlib
import subprocess
import time
import csv
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pipeline version for tracking
PIPELINE_VERSION = "2.1.0"

# Enhanced Data Collection Tracking Classes
class DataCollectionTracker:
    """Enhanced data collection tracking with file versioning"""
    
    def __init__(self, db_path=None):
        self.db_path = str(db_path) if db_path else str(Path(__file__).parent / "real_estate.db")
        self.current_run_id = None
        
    def get_file_hash(self, file_path):
        """Calculate SHA-256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def get_file_info(self, file_path):
        """Get comprehensive file information"""
        if not file_path or not Path(file_path).exists():
            return {}
            
        path = Path(file_path)
        return {
            "name": path.name,
            "size": path.stat().st_size,
            "hash": self.get_file_hash(path),
            "path": str(path.absolute())
        }
    
    def log_pipeline_stage(self, stage, source_file=None, output_file=None, 
                          records_processed=None, records_imported=None, 
                          parameters=None, status="completed", duration=None,
                          error_log=None, metadata=None):
        """Log a pipeline stage with comprehensive tracking"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get file information
        source_info = self.get_file_info(source_file) if source_file else {}
        output_info = self.get_file_info(output_file) if output_file else {}
        
        # Prepare parameters based on stage
        stage_params = {}
        if stage == "scraping":
            stage_params = {"scraping_parameters": json.dumps(parameters) if parameters else None}
        elif stage == "preprocessing":
            stage_params = {"preprocessing_parameters": json.dumps(parameters) if parameters else None}
        elif stage == "feature_engineering":
            stage_params = {
                "feature_engineering_version": "2.0.0",
                "feature_engineering_parameters": json.dumps(parameters) if parameters else None
            }
        
        cursor.execute("""
            INSERT INTO data_collection_history (
                collection_timestamp, pipeline_version, pipeline_stage,
                source_file_path, source_file_name, source_file_size, source_file_hash,
                output_file_path, output_file_name, output_file_size,
                records_processed, records_imported, processing_status,
                processing_duration_seconds, scraping_parameters, preprocessing_parameters,
                feature_engineering_version, feature_engineering_parameters,
                error_log, metadata, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            PIPELINE_VERSION,
            stage,
            source_info.get("path"),
            source_info.get("name"),
            source_info.get("size"),
            source_info.get("hash"),
            output_info.get("path"),
            output_info.get("name"),
            output_info.get("size"),
            records_processed,
            records_imported,
            status,
            duration,
            stage_params.get("scraping_parameters"),
            stage_params.get("preprocessing_parameters"),
            stage_params.get("feature_engineering_version"),
            stage_params.get("feature_engineering_parameters"),
            error_log,
            json.dumps(metadata) if metadata else None,
            "webapp_user"
        ))
        
        self.current_run_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Logged pipeline stage '{stage}' with ID {self.current_run_id}")
        return self.current_run_id

class SmartFileManager:
    """Smart file management for handling multiple CSV versions"""
    
    def __init__(self, tracker: DataCollectionTracker):
        self.tracker = tracker
    
    def get_latest_file(self, directory, pattern="*.csv", prefer_timestamp=True):
        """Get the most recent file matching pattern"""
        files = list(Path(directory).glob(pattern))
        if not files:
            return None
            
        if prefer_timestamp:
            # Try to extract timestamp from filename first
            timestamped_files = []
            for f in files:
                if any(ts in f.name for ts in ["20", "_202", "202"]):  # Look for year patterns
                    timestamped_files.append(f)
            
            if timestamped_files:
                # Sort by filename (timestamp should be in filename)
                return sorted(timestamped_files, key=lambda x: x.name)[-1]
        
        # Fallback to modification time
        return max(files, key=lambda x: x.stat().st_mtime)

# Security configuration
SECRET_KEY = "your-secret-key-change-this-in-production-2024-enhanced"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)

# Initialize FastAPI app
app = FastAPI(
    title="Tajikistan Real Estate Analytics Platform - Full Integration",
    description="Complete platform with authentication, ML predictions, bargain detection, and user management",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
db_path = Path(__file__).parent / "real_estate.db"
model = None
model_metadata = None
properties_df = None
rental_predictor = None

# Import rental prediction system
try:
    import sys
    import os
    utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
    sys.path.append(utils_path)
    from predict_rental import RentalPricePredictor
    logger.info(f"Successfully imported RentalPricePredictor from {utils_path}")
except ImportError as e:
    logger.warning(f"Could not import rental predictor: {e}")
    RentalPricePredictor = None

# Pydantic Models
class UserBase(BaseModel):
    email: EmailStr
    username: Optional[str] = None  # Optional since we use email as identifier
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserInDB(UserBase):
    id: int
    hashed_password: str
    is_active: bool = True
    has_collected_data: bool = False  # Track if user has collected data
    created_at: datetime
    email_notifications: bool = True
    push_notifications: bool = True
    notification_frequency: str = "daily"

class User(UserBase):
    id: int
    is_active: bool
    has_collected_data: bool = False  # Track if user has collected data
    created_at: datetime
    email_notifications: bool
    push_notifications: bool
    notification_frequency: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: User

class PropertyResponse(BaseModel):
    id: int
    title: Optional[str]
    url: str
    price: float
    price_per_sqm: Optional[float]
    area: Optional[float]
    rooms: Optional[int]
    floor: Optional[int]
    total_floors: Optional[int]
    city: Optional[str]
    district: Optional[str]
    address: Optional[str]
    build_state: Optional[str]
    property_type: Optional[str]
    renovation: Optional[str] = None
    image_urls: Optional[List[str]] = None
    predicted_price: Optional[float]
    price_difference: Optional[float]
    price_difference_percentage: Optional[float]
    bargain_score: Optional[float]
    bargain_category: Optional[str]
    renovation_category: Optional[str] = None
    global_bargain_category: Optional[str] = None
    is_favorite: bool = False
    view_count: int = 0
    # Investment analysis fields
    estimated_monthly_rent: Optional[float] = None
    annual_rental_income: Optional[float] = None
    gross_rental_yield: Optional[float] = None
    net_rental_yield: Optional[float] = None
    roi_percentage: Optional[float] = None
    payback_period_years: Optional[float] = None
    monthly_cash_flow: Optional[float] = None
    investment_category: Optional[str] = None
    cash_flow_category: Optional[str] = None
    rental_prediction_confidence: Optional[float] = None
    # NEW: Renovation cost analysis
    estimated_renovation_cost: Optional[float] = None
    renovation_cost_with_buffer: Optional[float] = None
    total_investment_required: Optional[float] = None
    renovation_percentage_of_price: Optional[float] = None
    # NEW: Rental premium for renovations
    monthly_rent_premium: Optional[float] = None
    annual_rent_premium: Optional[float] = None
    renovation_premium_multiplier: Optional[float] = None
    renovation_roi_annual: Optional[float] = None
    # NEW: Risk assessment
    overall_risk_score: Optional[float] = None
    risk_category: Optional[str] = None
    renovation_complexity_risk: Optional[float] = None
    financial_risk: Optional[float] = None
    market_risk: Optional[float] = None
    execution_risk: Optional[float] = None
    # NEW: Final recommendations
    final_investment_recommendation: Optional[str] = None
    investment_priority_score: Optional[float] = None
    investment_priority_category: Optional[str] = None
    # NEW: Investment flags
    is_premium_district: Optional[bool] = None
    has_high_renovation_roi: Optional[bool] = None
    is_fast_payback: Optional[bool] = None
    has_significant_premium: Optional[bool] = None

class PredictionRequest(BaseModel):
    rooms: Optional[int] = Field(3, ge=1, le=10, description="Number of rooms")
    area_m2: float = Field(..., gt=0, description="Property area in square meters")
    floor: Optional[int] = Field(None, ge=1, le=50, description="Floor number")
    district: Optional[str] = Field(None, description="District name (e.g., 'Душанбе')")
    renovation: Optional[str] = Field(None, description="Renovation state (e.g., 'Без ремонта (коробка)', 'С ремонтом', 'Новый ремонт')")
    bathroom: Optional[str] = Field(None, description="Bathroom type (e.g., 'Раздельный', 'Совмещенный')")
    heating: Optional[str] = Field(None, description="Heating type (e.g., 'Нет', 'Есть')")
    price_per_m2: Optional[float] = Field(None, description="Property price per square meter for more accurate prediction")

class PredictionResponse(BaseModel):
    predicted_rental: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    annual_rental_income: float
    gross_rental_yield: float
    features_used: Dict[str, Any] = {}
    model_info: Optional[Dict[str, Any]] = None

class FavoriteRequest(BaseModel):
    property_id: int
    notes: Optional[str] = None

class AlertRequest(BaseModel):
    alert_name: str
    alert_type: str = Field(..., pattern="^(bargain|price_drop|new_listing)$")
    conditions: Dict[str, Any]

class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None

class DashboardStats(BaseModel):
    total_properties: int
    total_bargains: int
    excellent_bargains: int
    good_bargains: int
    user_favorites: int
    user_searches: int
    user_predictions: int
    active_alerts: int
    avg_savings_percentage: float

class FilterRanges(BaseModel):
    price_min: float
    price_max: float
    area_min: float
    area_max: float
    floor_min: float
    floor_max: float

class MarketStatsResponse(BaseModel):
    total_listings: int
    avg_price: float
    median_price: float
    min_price: float
    max_price: float
    avg_price_per_sqm: float
    total_bargains: int
    excellent_bargains: int
    good_bargains: int
    price_distribution: Dict[str, Any]
    room_distribution: Dict[str, Any]
    city_distribution: Dict[str, Any]

class ScrapingRequest(BaseModel):
    rooms: str = Field(..., description="Number of rooms (e.g., '3-komnatnyie')")
    city: str = Field(..., description="City name (e.g., 'hudzhand')")
    build_state: str = Field(default="sostoyanie---10", description="Build state (e.g., 'sostoyanie---10' for built)")
    property_type: Optional[str] = Field(None, description="Property type (e.g., 'type---1' for secondary market)")

class ScrapingResponse(BaseModel):
    status: str
    message: str
    task_id: Optional[str] = None
    total_scraped: Optional[int] = None
    processing_status: str
    files_created: List[str] = []

class PipelineStatsResponse(BaseModel):
    total_runs: int
    successful_runs: int
    failed_runs: int
    latest_run: Optional[Dict[str, Any]]
    pipeline_version: str
    recent_runs: List[Dict[str, Any]]

class DataCollectionHistoryResponse(BaseModel):
    id: int
    collection_timestamp: str
    pipeline_stage: str
    processing_status: str
    records_processed: Optional[int]
    records_imported: Optional[int]
    processing_duration_seconds: Optional[float]
    source_file_name: Optional[str]
    output_file_name: Optional[str]

# Database Helper Functions
def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

def get_user_by_email(email: str) -> Optional[UserInDB]:
    """Get user by email from database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        # Helper function to safely get column value
        def safe_get_column(row, column_name, default=None):
            try:
                return row[column_name] if row[column_name] is not None else default
            except (IndexError, KeyError):
                return default
        
        return UserInDB(
            id=row['id'],
            email=row['email'],
            username=safe_get_column(row, 'username') or row['email'],  # Use username or fallback to email
            full_name=row['full_name'],
            hashed_password=row['hashed_password'],
            is_active=bool(row['is_active']),
            has_collected_data=bool(safe_get_column(row, 'has_collected_data', False)),
            created_at=datetime.fromisoformat(row['created_at']),
            email_notifications=bool(safe_get_column(row, 'notification_enabled', True)),
            push_notifications=bool(safe_get_column(row, 'push_notifications', True)),
            notification_frequency=safe_get_column(row, 'notification_frequency', 'daily')
        )
    return None

def create_user_in_db(user_data: UserCreate) -> UserInDB:
    """Create new user in database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    hashed_password = pwd_context.hash(user_data.password)
    
    cursor.execute("""
        INSERT INTO users (email, username, full_name, hashed_password, created_at, is_active, is_verified, notification_enabled, has_collected_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_data.email, user_data.email, user_data.full_name, 
          hashed_password, datetime.utcnow().isoformat(), True, False, True, False))
    
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return UserInDB(
        id=user_id,
        email=user_data.email,
        username=user_data.email,  # Use email as username
        full_name=user_data.full_name,
        hashed_password=hashed_password,
        is_active=True,
        has_collected_data=False,
        created_at=datetime.utcnow(),
        email_notifications=True,
        push_notifications=True,
        notification_frequency="daily"
    )

# Authentication Functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    user = get_user_by_email(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user_by_email(email=username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_user_optional(token: Optional[str] = Depends(oauth2_scheme_optional)) -> Optional[UserInDB]:
    """Get current user if token is provided, otherwise return None"""
    if not token:
        return None
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            return None
    except JWTError:
        return None
    
    user = get_user_by_email(email=username)
    return user

# Rental Model Loading
def load_rental_model():
    """Load the trained XGBoost rental prediction model"""
    global rental_predictor
    
    try:
        if RentalPricePredictor is not None:
            rental_predictor = RentalPricePredictor()
            logger.info("Rental prediction model loaded successfully")
        else:
            logger.warning("RentalPricePredictor class not available")
    except Exception as e:
        logger.error(f"Error loading rental prediction model: {e}")
        rental_predictor = None



def load_properties_data():
    """Load properties data from CSV as fallback when database is empty"""
    global properties_df
    
    try:
        # Define potential paths for the data
        somon_project_dir = Path(__file__).parent.parent.parent
        csv_paths = [
            somon_project_dir / "data" / "feature_engineered" / "listings_with_features.csv",
            somon_project_dir / "data" / "preprocessed" / "cleaned_listings_v2.csv",
            Path(__file__).parent / "listings_with_features.csv",
            Path(__file__).parent / "cleaned_listings_v2.csv"
        ]
        
        for csv_path in csv_paths:
            if csv_path.exists():
                try:
                    properties_df = pd.read_csv(csv_path)
                    
                    # Also try to load raw data for image URLs if available
                    try:
                        raw_data_dir = somon_project_dir / "data" / "raw"
                        raw_files = list(raw_data_dir.glob("scraped_listings_*.csv"))
                        if raw_files:
                            # Sort by filename to get the latest
                            latest_raw_file = sorted(raw_files)[-1]
                            raw_df = pd.read_csv(latest_raw_file)
                            logger.info(f"Loading image URLs from: {latest_raw_file.name}")
                            
                            # Merge image URLs if possible
                            if 'url' in properties_df.columns and 'url' in raw_df.columns:
                                properties_df = properties_df.merge(raw_df[['url', 'image_urls']], on='url', how='left')
                    except Exception as e:
                        logger.warning(f"Could not load raw data for image URLs: {e}")
                    
                    logger.info(f"Loaded {len(properties_df)} properties from {csv_path}")
                    
                    # Log data collection event
                    # Get bargain statistics if available
                    bargain_stats = {}
                    if 'bargain_category' in properties_df.columns:
                        bargain_counts = properties_df['bargain_category'].value_counts().to_dict()
                        bargain_stats = {
                            "total_with_bargain_data": len(properties_df[properties_df['bargain_category'].notna()]),
                            "bargain_breakdown": bargain_counts
                        }
                    
                    logger.info(f"Loaded properties data from {csv_path.name}: {len(properties_df)} properties")
                    if bargain_stats:
                        logger.info(f"Bargain statistics: {bargain_stats}")
                    
                    return
                except Exception as e:
                    logger.warning(f"Could not load from {csv_path}: {e}")
                    continue
        
        logger.warning("No property data source found")
        properties_df = pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error loading property data: {e}")
        properties_df = pd.DataFrame()

def import_value_scores():
    """Import value_score from CSV into database investment_score column"""
    
    # Paths
    somon_project_dir = Path(__file__).parent.parent.parent
    csv_path = somon_project_dir / "data" / "feature_engineered" / "listings_with_features.csv"
    
    # Check if CSV exists
    if not csv_path.exists():
        logger.warning(f"CSV file not found for value scores: {csv_path}")
        return False
    
    try:
        # Load CSV data
        logger.info(f"Loading value scores from {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records from CSV")
        
        # Check required columns
        required_cols = ['url', 'value_score']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in CSV: {missing_cols}")
            return False
        
        # Connect to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get existing properties
        cursor.execute("SELECT id, url FROM property_listings")
        db_properties = {row[1]: row[0] for row in cursor.fetchall()}
        logger.info(f"Found {len(db_properties)} properties in database")
        
        # Update properties with value scores
        updated_count = 0
        not_found_count = 0
        
        for _, row in df.iterrows():
            url = row['url']
            value_score = row['value_score']
            
            # Skip if value_score is NaN
            if pd.isna(value_score):
                continue
                
            if url in db_properties:
                property_id = db_properties[url]
                
                # Also get other available scores
                bargain_score = row.get('bargain_score') if 'bargain_score' in row and pd.notna(row.get('bargain_score')) else None
                bargain_category = row.get('bargain_category') if 'bargain_category' in row and pd.notna(row.get('bargain_category')) else None
                predicted_price = row.get('predicted_price') if 'predicted_price' in row and pd.notna(row.get('predicted_price')) else None
                publication_weekday = row.get('publication_weekday') if 'publication_weekday' in row and pd.notna(row.get('publication_weekday')) else None
                
                # Update the property with scores and publication weekday
                cursor.execute("""
                    UPDATE property_listings 
                    SET investment_score = ?, 
                        bargain_score = ?,
                        bargain_category = ?,
                        predicted_price = ?,
                        publication_weekday = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (value_score, bargain_score, bargain_category, predicted_price, publication_weekday, property_id))
                
                updated_count += 1
            else:
                not_found_count += 1
        
        # Commit changes
        conn.commit()
        
        # Verify the update
        cursor.execute("SELECT COUNT(*) FROM property_listings WHERE investment_score IS NOT NULL")
        updated_properties = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(investment_score), MIN(investment_score), MAX(investment_score) FROM property_listings WHERE investment_score IS NOT NULL")
        stats = cursor.fetchone()
        
        conn.close()
        
        logger.info(f"Successfully updated {updated_count} properties with investment scores")
        logger.info(f"Properties not found in database: {not_found_count}")
        logger.info(f"Total properties with investment_score: {updated_properties}")
        if stats[0] is not None:
            logger.info(f"Investment score stats - Avg: {stats[0]:.4f}, Min: {stats[1]:.4f}, Max: {stats[2]:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error importing value scores: {e}")
        return False

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting Tajikistan Real Estate Platform - Full Integration v2.0.0")
    
    # Initialize database tables if they don't exist
    try:
        logger.info("Checking database...")
        # Just ensure we can connect - don't force table creation on startup
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if basic tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        users_table_exists = cursor.fetchone() is not None
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='property_listings'")
        properties_table_exists = cursor.fetchone() is not None
        
        conn.close()
        
        if not users_table_exists or not properties_table_exists:
            logger.info("Creating missing database tables...")
            try:
                from init_database import create_database_tables
                create_database_tables()
                logger.info("Database tables created successfully")
            except Exception as e:
                logger.warning(f"Could not create database tables: {e}")
        else:
            logger.info("Database tables exist")
            
    except Exception as e:
        logger.warning(f"Database check failed: {e}")
        logger.info("Will continue without database validation")
    
    load_rental_model()
    load_properties_data()  # Add data loading
    logger.info("Application startup complete")

# Health Check Endpoint
@app.get("/debug/path")
async def debug_path():
    """Debug endpoint to check paths"""
    import os
    return {
        "current_working_directory": os.getcwd(),
        "db_path": db_path,
        "absolute_db_path": os.path.abspath(db_path),
        "db_exists": os.path.exists(db_path),
        "db_size": os.path.getsize(db_path) if os.path.exists(db_path) else 0
    }

@app.get("/health")
async def health_check():
    """Simple health check for Railway"""
    return {"status": "healthy", "service": "somongpt-backend"}

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with all components"""
    try:
        # Quick database check
        db_status = "unknown"
        model_status = "unknown"
        
        try:
            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()
            
            # Check if properties table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='properties'
            """)
            
            if cursor.fetchone():
                # Quick count check
                cursor.execute("SELECT COUNT(*) FROM properties")
                property_count = cursor.fetchone()[0]
                db_status = f"connected ({property_count} properties)"
            else:
                db_status = "connected (no properties table)"
                
            conn.close()
        except Exception as e:
            db_status = f"error: {str(e)[:50]}"
        
        # Check ML model
        try:
            global model, model_loaded, model_metadata
            if model_loaded and model is not None:
                model_status = f"loaded (accuracy: {model_metadata.get('accuracy', 'unknown')})"
            else:
                model_status = "not loaded"
        except Exception as e:
            model_status = f"error: {str(e)[:50]}"
        
        return {
            "status": "healthy",
            "service": "somongpt-backend",
            "database": db_status,
            "ml_model": model_status,
            "environment": os.getenv("ENVIRONMENT", "development"),
            "python_path": os.getenv("PYTHONPATH", "not set")
        }
        
    except Exception as e:
        return {
            "status": "error",
            "service": "somongpt-backend", 
            "error": str(e)
        }

@app.get("/")
async def health_check():
    """Detailed health check endpoint"""
    try:
        # Test database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if tables exist first
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='property_listings'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            cursor.execute("SELECT COUNT(*) FROM property_listings")
            total_properties = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(price) FROM property_listings WHERE price IS NOT NULL")
            avg_price_result = cursor.fetchone()
            avg_price = float(avg_price_result[0]) if avg_price_result[0] else 0
            
            cursor.execute("SELECT COUNT(*) FROM property_listings WHERE bargain_category IN ('excellent_bargain', 'good_bargain')")
            investment_opportunities = cursor.fetchone()[0]
        else:
            total_properties = 0
            avg_price = 0
            investment_opportunities = 0
        
        conn.close()
        
        return {
            "status": "healthy",
            "model_loaded": rental_predictor is not None,
            "database_connected": True,
            "table_exists": table_exists,
            "total_properties": total_properties,
            "avg_price": avg_price,
            "investment_opportunities": investment_opportunities,
            "model_accuracy": 0.721 if rental_predictor else None,
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "model_loaded": rental_predictor is not None,
            "database_connected": False,
            "table_exists": False,
            "total_properties": 0,
            "avg_price": 0,
            "investment_opportunities": 0,
            "model_accuracy": None,
            "last_updated": datetime.now().isoformat(),
            "error": str(e)
        }

# Authentication Routes
@app.post("/auth/register", response_model=ApiResponse)
async def register_user(user_data: UserCreate, background_tasks: BackgroundTasks):
    """Register a new user"""
    if get_user_by_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    user = create_user_in_db(user_data)
    logger.info(f"New user registered: {user.email}")
    
    # Send welcome email in the background
    def send_welcome_email_task():
        try:
            logger.info(f"🔄 Starting welcome email task for {user.email}")
            
            # Import with proper path
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, current_dir)
            
            from services.notification_service import EmailNotificationService
            
            logger.info("📧 EmailNotificationService imported successfully")
            email_service = EmailNotificationService()
            
            logger.info(f"📧 Email service configured: {email_service.is_configured}")
            
            # Use full_name if available, otherwise use email prefix
            display_name = user.full_name or user.email.split('@')[0]
            logger.info(f"📧 Sending welcome email to {user.email} (display name: {display_name})")
            
            success = email_service.send_welcome_email(user.email, display_name)
            if success:
                logger.info(f"✅ Welcome email sent successfully to {user.email}")
            else:
                logger.warning(f"⚠️  Failed to send welcome email to {user.email} - Email service may not be configured")
        except ImportError as e:
            logger.error(f"❌ Import error for welcome email to {user.email}: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Error sending welcome email to {user.email}: {str(e)}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
    
    # Add the welcome email task to background tasks
    background_tasks.add_task(send_welcome_email_task)
    
    return ApiResponse(
        success=True,
        message="User registered successfully! Check your email for a welcome message.",
        data={"user_id": user.id, "email": user.email}
    )

@app.post("/auth/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and get access token"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    logger.info(f"User logged in: {user.username}")
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=User(
            id=user.id,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            is_active=user.is_active,
            has_collected_data=user.has_collected_data,
            created_at=user.created_at,
            email_notifications=user.email_notifications,
            push_notifications=user.push_notifications,
            notification_frequency=user.notification_frequency
        )
    )

@app.get("/auth/me", response_model=User)
async def read_users_me(current_user: UserInDB = Depends(get_current_active_user)):
    """Get current user profile"""
    return User(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        has_collected_data=current_user.has_collected_data,
        created_at=current_user.created_at,
        email_notifications=current_user.email_notifications,
        push_notifications=current_user.push_notifications,
        notification_frequency=current_user.notification_frequency
    )

# Property Routes
@app.get("/properties", response_model=List[PropertyResponse])
async def search_properties(
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    min_area: Optional[float] = Query(None),
    max_area: Optional[float] = Query(None),
    min_rooms: Optional[int] = Query(None),
    max_rooms: Optional[int] = Query(None),
    cities: Optional[str] = Query(None, description="Comma-separated list of cities"),
    districts: Optional[str] = Query(None, description="Comma-separated list of districts"),
    build_states: Optional[str] = Query(None, description="Comma-separated list of build states"),
    renovations: Optional[str] = Query(None, description="Comma-separated list of renovation states"),
    min_floor: Optional[int] = Query(None),
    max_floor: Optional[int] = Query(None),
    bargain_categories: Optional[str] = Query(None, description="Comma-separated list: excellent,good,fair"),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    current_user: Optional[UserInDB] = Depends(get_current_user_optional)
):
    """Search properties with comprehensive filtering options"""
    
    # Add logging to debug filter parameters
    logger.info(f"Search parameters: min_price={min_price}, max_price={max_price}, "
                f"cities={cities}, districts={districts}, build_states={build_states}, "
                f"renovations={renovations}, bargain_categories={bargain_categories}")
    
    try:
        # Try database first
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build query with filters - only show properties collected by current user
        if current_user:
            query = "SELECT * FROM property_listings WHERE collected_by_user_id = ?"
            params = [current_user.id]
            logger.info(f"Starting database query for properties collected by user {current_user.id}")
        else:
            # If no user logged in, return empty results (user-specific properties only)
            logger.info("No user authenticated, returning empty results (user-specific properties only)")
            return []
        
        if min_price:
            query += " AND price >= ?"
            params.append(min_price)
        if max_price:
            query += " AND price <= ?"
            params.append(max_price)
        if min_area:
            query += " AND area >= ?"
            params.append(min_area)
        if max_area:
            query += " AND area <= ?"
            params.append(max_area)
        if min_rooms:
            query += " AND rooms >= ?"
            params.append(min_rooms)
        if max_rooms:
            query += " AND rooms <= ?"
            params.append(max_rooms)
        if min_floor:
            query += " AND floor >= ?"
            params.append(min_floor)
        if max_floor:
            query += " AND floor <= ?"
            params.append(max_floor)
            
        # Handle list filters
        if cities:
            city_list = [c.strip() for c in cities.split(',')]
            placeholders = ','.join(['?' for _ in city_list])
            query += f" AND city IN ({placeholders})"
            params.extend(city_list)
            
        if districts:
            district_list = [d.strip() for d in districts.split(',')]
            placeholders = ','.join(['?' for _ in district_list])
            query += f" AND district IN ({placeholders})"
            params.extend(district_list)
            
        if build_states:
            build_state_list = [b.strip() for b in build_states.split(',')]
            logger.info(f"Frontend build_state filter: {build_state_list}")
            
            # Map frontend build state terms to database values (Новостройка/Вторичный рынок)
            build_state_mapping = {
                # English terms
                'new_construction': 'Новостройка',
                'secondary_market': 'Вторичный рынок',
                'primary': 'Новостройка',
                'secondary': 'Вторичный рынок',
                # Direct Russian mappings
                'Новостройка': 'Новостройка',
                'Вторичный рынок': 'Вторичный рынок'
            }
            
            mapped_build_states = []
            for build_state in build_state_list:
                if build_state in build_state_mapping:
                    mapped_build_states.append(build_state_mapping[build_state])
                    logger.info(f"Mapped build_state '{build_state}' to '{build_state_mapping[build_state]}'")
                else:
                    # If no mapping found, try direct match
                    mapped_build_states.append(build_state)
                    logger.info(f"Direct mapping for build_state '{build_state}'")
            
            if mapped_build_states:
                # Remove duplicates while preserving order
                unique_build_states = list(dict.fromkeys(mapped_build_states))
                placeholders = ','.join(['?' for _ in unique_build_states])
                query += f" AND build_state IN ({placeholders})"
                params.extend(unique_build_states)
                logger.info(f"Final build_state filter: {unique_build_states}")
            
        # Handle renovations parameter - map to renovation column  
        if renovations:
            renovation_list = [r.strip() for r in renovations.split(',')]
            logger.info(f"Frontend renovation filter: {renovation_list}")
            
            # Map frontend renovation terms to database renovation values
            # renovation contains: "Новый ремонт", "С ремонтом", "Без ремонта (коробка)"
            renovation_mapping = {
                # English terms
                'new_renovation': 'Новый ремонт',
                'with_renovation': 'С ремонтом', 
                'no_renovation': 'Без ремонта (коробка)',
                'shell_finish': 'Без ремонта (коробка)',
                'renovated': 'С ремонтом',
                'needs_renovation': 'Без ремонта (коробка)',
                # Russian terms that frontend might send
                'Новый ремонт': 'Новый ремонт',
                'С ремонтом': 'С ремонтом',
                'Без ремонта (коробка)': 'Без ремонта (коробка)',
                'Без ремонта': 'Без ремонта (коробка)'
            }
            
            mapped_renovations = []
            for renovation in renovation_list:
                if renovation in renovation_mapping:
                    mapped_renovations.append(renovation_mapping[renovation])
                    logger.info(f"Mapped renovation '{renovation}' to '{renovation_mapping[renovation]}'")
                else:
                    # If no mapping found, try direct match
                    mapped_renovations.append(renovation)
                    logger.info(f"Direct mapping for renovation '{renovation}'")
            
            if mapped_renovations:
                # Remove duplicates while preserving order
                unique_mapped_renovations = list(dict.fromkeys(mapped_renovations))
                placeholders = ','.join(['?' for _ in unique_mapped_renovations])
                query += f" AND renovation IN ({placeholders})"
                params.extend(unique_mapped_renovations)
                logger.info(f"Final renovation filter: {unique_mapped_renovations}")
                
        if bargain_categories:
            bargain_list = [b.strip() for b in bargain_categories.split(',')]
            logger.info(f"Frontend bargain_categories filter: {bargain_list}")
            
            # Map frontend bargain category terms to database values
            bargain_category_mapping = {
                'excellent': 'excellent_bargain',  # Fixed: map to excellent_bargain (23 properties)
                'exceptional': 'exceptional_opportunity',  # Keep rare exceptional (1 property)
                'good': 'good_bargain', 
                'fair': 'fair_value',
                'market': 'market_price',
                'overpriced': 'overpriced',
                # Direct mappings for database values
                'exceptional_opportunity': 'exceptional_opportunity',
                'excellent_bargain': 'excellent_bargain',  # Fixed: direct mapping
                'good_bargain': 'good_bargain',
                'fair_value': 'fair_value',
                'market_price': 'market_price'
            }
            
            mapped_bargain_categories = []
            for category in bargain_list:
                if category in bargain_category_mapping:
                    mapped_bargain_categories.append(bargain_category_mapping[category])
                    logger.info(f"Mapped bargain_category '{category}' to '{bargain_category_mapping[category]}'")
                else:
                    # If no mapping found, try direct match
                    mapped_bargain_categories.append(category)
                    logger.info(f"Direct mapping for bargain_category '{category}'")
            
            if mapped_bargain_categories:
                # Remove duplicates while preserving order
                unique_bargain_categories = list(dict.fromkeys(mapped_bargain_categories))
                placeholders = ','.join(['?' for _ in unique_bargain_categories])
                query += f" AND bargain_category IN ({placeholders})"
                params.extend(unique_bargain_categories)
                logger.info(f"Final bargain_category filter: {unique_bargain_categories}")
        
        # Add ordering and pagination
        query += " ORDER BY price ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        logger.info(f"Database query returned {len(rows)} rows")
        
        # Get user favorites if authenticated
        user_favorites = set()
        if current_user:
            fav_cursor = conn.cursor()
            fav_cursor.execute(
                "SELECT property_id FROM user_favorites WHERE user_id = ?",
                (current_user.id,)
            )
            user_favorites = {row[0] for row in fav_cursor.fetchall()}
            
            # Log user search
            search_params = {
                "min_price": min_price, "max_price": max_price,
                "min_area": min_area, "max_area": max_area,
                "cities": cities, "districts": districts,
                "build_states": build_states, "bargain_categories": bargain_categories,
                "limit": limit, "offset": offset
            }
            
            cursor.execute("""
                INSERT INTO user_searches (user_id, search_type, search_params, results_count)
                VALUES (?, ?, ?, ?)
            """, (current_user.id, "property_search", json.dumps(search_params), len(rows)))
            conn.commit()
        
        properties = []
        for row in rows:
            try:
                # Helper function to safely get column value from sqlite3.Row
                def safe_get_row_value(row, column_name, default=None):
                    try:
                        return row[column_name] if row[column_name] is not None else default
                    except (IndexError, KeyError):
                        return default
                
                # Handle image_urls - convert from JSON string to list
                image_urls = None
                image_urls_raw = safe_get_row_value(row, 'image_urls')
                if image_urls_raw:
                    try:
                        if isinstance(image_urls_raw, str):
                            # If it's a semicolon-separated string, split it
                            if ';' in image_urls_raw:
                                image_urls = [url.strip() for url in image_urls_raw.split(';') if url.strip()]
                            else:
                                # Try to parse as JSON
                                image_urls = json.loads(image_urls_raw)
                        elif isinstance(image_urls_raw, list):
                            image_urls = image_urls_raw
                    except:
                        image_urls = None
            
                properties.append(PropertyResponse(
                    id=row['id'],
                    title=safe_get_row_value(row, 'title'),
                    url=safe_get_row_value(row, 'url'),
                    price=row['price'],
                    price_per_sqm=safe_get_row_value(row, 'price_per_sqm'),
                    area=safe_get_row_value(row, 'area'),
                    rooms=safe_get_row_value(row, 'rooms'),
                    floor=safe_get_row_value(row, 'floor'),
                    total_floors=safe_get_row_value(row, 'total_floors'),
                    city=safe_get_row_value(row, 'city'),
                    district=safe_get_row_value(row, 'district'),
                    address=safe_get_row_value(row, 'address'),
                    build_state=safe_get_row_value(row, 'build_state'),
                    property_type=safe_get_row_value(row, 'property_type'),
                    renovation=safe_get_row_value(row, 'renovation'),
                    image_urls=image_urls,
                    predicted_price=safe_get_row_value(row, 'predicted_price'),
                    price_difference=safe_get_row_value(row, 'price_difference'),
                    price_difference_percentage=safe_get_row_value(row, 'price_difference_percentage'),
                    bargain_score=safe_get_row_value(row, 'bargain_score'),
                    bargain_category=safe_get_row_value(row, 'enhanced_bargain_category') or safe_get_row_value(row, 'bargain_category'),  # Prefer enhanced category
                    renovation_category=safe_get_row_value(row, 'renovation_category'),
                    global_bargain_category=safe_get_row_value(row, 'global_bargain_category'),
                    is_favorite=row['id'] in user_favorites,
                    view_count=safe_get_row_value(row, 'view_count', 0),
                    # Investment analysis fields
                    estimated_monthly_rent=safe_get_row_value(row, 'estimated_monthly_rent'),
                    annual_rental_income=safe_get_row_value(row, 'annual_rental_income'),
                    gross_rental_yield=safe_get_row_value(row, 'gross_rental_yield'),
                    net_rental_yield=safe_get_row_value(row, 'net_rental_yield'),
                    roi_percentage=safe_get_row_value(row, 'roi_percentage'),
                    payback_period_years=safe_get_row_value(row, 'payback_period_years'),
                    monthly_cash_flow=safe_get_row_value(row, 'monthly_cash_flow'),
                    investment_category=safe_get_row_value(row, 'investment_category'),
                    cash_flow_category=safe_get_row_value(row, 'cash_flow_category'),
                    rental_prediction_confidence=safe_get_row_value(row, 'rental_prediction_confidence'),
                    # NEW: Renovation cost analysis
                    estimated_renovation_cost=safe_get_row_value(row, 'estimated_renovation_cost'),
                    renovation_cost_with_buffer=safe_get_row_value(row, 'renovation_cost_with_buffer'),
                    total_investment_required=safe_get_row_value(row, 'total_investment_required'),
                    renovation_percentage_of_price=safe_get_row_value(row, 'renovation_percentage_of_price'),
                    # NEW: Rental premium for renovations
                    monthly_rent_premium=safe_get_row_value(row, 'monthly_rent_premium'),
                    annual_rent_premium=safe_get_row_value(row, 'annual_rent_premium'),
                    renovation_premium_multiplier=safe_get_row_value(row, 'renovation_premium_multiplier'),
                    renovation_roi_annual=safe_get_row_value(row, 'renovation_roi_annual'),
                    # NEW: Risk assessment
                    overall_risk_score=safe_get_row_value(row, 'overall_risk_score'),
                    risk_category=safe_get_row_value(row, 'risk_category'),
                    renovation_complexity_risk=safe_get_row_value(row, 'renovation_complexity_risk'),
                    financial_risk=safe_get_row_value(row, 'financial_risk'),
                    market_risk=safe_get_row_value(row, 'market_risk'),
                    execution_risk=safe_get_row_value(row, 'execution_risk'),
                    # NEW: Final recommendations
                    final_investment_recommendation=safe_get_row_value(row, 'final_investment_recommendation'),
                    investment_priority_score=safe_get_row_value(row, 'investment_priority_score'),
                    investment_priority_category=safe_get_row_value(row, 'investment_priority_category'),
                    # NEW: Investment flags
                    is_premium_district=safe_get_row_value(row, 'is_premium_district'),
                    has_high_renovation_roi=safe_get_row_value(row, 'has_high_renovation_roi'),
                    is_fast_payback=safe_get_row_value(row, 'is_fast_payback'),
                    has_significant_premium=safe_get_row_value(row, 'has_significant_premium')
                ))
            except Exception as e:
                logger.error(f"Error processing database row {row.get('id', 'unknown')}: {e}")
                continue  # Skip this row and continue with the next one
        
        # Always check database first and prefer it over CSV
        if properties:
            conn.close()
            logger.info(f"Returning {len(properties)} properties from database")
            return properties
        
        # Log that database was empty
        logger.info("Database returned no results, checking if database has any properties at all")
        cursor.execute("SELECT COUNT(*) FROM property_listings")
        total_db_properties = cursor.fetchone()[0]
        logger.info(f"Total properties in database: {total_db_properties}")
        conn.close()
        
        # Only fallback to CSV data if database is completely empty
        if total_db_properties == 0 and properties_df is not None and not properties_df.empty:
            logger.info("Database is empty, falling back to CSV data")
            df = properties_df.copy()
            
            # Apply filters
            if min_price:
                df = df[df['price'] >= min_price]
            if max_price:
                df = df[df['price'] <= max_price]
            if min_area and 'area_m2' in df.columns:
                df = df[df['area_m2'] >= min_area]
            if max_area and 'area_m2' in df.columns:
                df = df[df['area_m2'] <= max_area]
            if min_rooms and 'rooms' in df.columns:
                df = df[df['rooms'] >= min_rooms]
            if max_rooms and 'rooms' in df.columns:
                df = df[df['rooms'] <= max_rooms]
            if min_floor and 'floor' in df.columns:
                df = df[df['floor'] >= min_floor]
            if max_floor and 'floor' in df.columns:
                df = df[df['floor'] <= max_floor]
            
            # Handle list filters
            if cities and 'city' in df.columns:
                city_list = [c.strip() for c in cities.split(',')]
                df = df[df['city'].isin(city_list)]
            
            if districts and 'district' in df.columns:
                district_list = [d.strip() for d in districts.split(',')]
                df = df[df['district'].isin(district_list)]
            
            if build_states and 'build_type' in df.columns:
                build_state_list = [b.strip() for b in build_states.split(',')]
                df = df[df['build_type'].isin(build_state_list)]
            
            if renovations and 'renovation' in df.columns:
                renovation_list = [r.strip() for r in renovations.split(',')]
                df = df[df['renovation'].isin(renovation_list)]
            
            if bargain_categories and 'bargain_category' in df.columns:
                bargain_list = [b.strip() for b in bargain_categories.split(',')]
                df = df[df['bargain_category'].isin(bargain_list)]
            
            # Apply pagination
            df = df.iloc[offset:offset + limit]
            
            # Convert to response format
            csv_properties = []
            for _, row in df.iterrows():
                # Process image URLs
                image_urls = []
                if 'image_urls' in row and pd.notna(row['image_urls']):
                    try:
                        if isinstance(row['image_urls'], str):
                            if row['image_urls'].startswith('['):
                                image_urls = json.loads(row['image_urls'].replace("'", '"'))
                        else:
                            # Split by semicolon first, then by comma as fallback
                            if ';' in row['image_urls']:
                                image_urls = [url.strip() for url in row['image_urls'].split(';') if url.strip()]
                            else:
                                image_urls = [url.strip() for url in row['image_urls'].split(',') if url.strip()]
                    except:
                        image_urls = []
                
                csv_properties.append(PropertyResponse(
                    id=0,  # CSV fallback doesn't have real IDs
                    title=row.get('title') if hasattr(row, 'get') else None,
                    url=row.get('url') if hasattr(row, 'get') else None,
                    price=float(row['price']) if pd.notna(row['price']) else 0,
                    price_per_sqm=float(row['price_per_sqm']) if 'price_per_sqm' in row and pd.notna(row['price_per_sqm']) else None,
                    rooms=int(row['rooms']) if 'rooms' in row and pd.notna(row['rooms']) else None,
                    area=float(row['area_m2']) if 'area_m2' in row and pd.notna(row['area_m2']) else None,
                    floor=int(row['floor']) if 'floor' in row and pd.notna(row['floor']) else None,
                    total_floors=int(row['total_floors']) if 'total_floors' in row and pd.notna(row['total_floors']) else None,
                    city=row.get('city') if hasattr(row, 'get') else None,
                    district=row.get('district') if hasattr(row, 'get') else None,
                    address=row.get('address') if hasattr(row, 'get') else None,
                    build_state=row.get('build_state') if hasattr(row, 'get') else None,
                    property_type=row.get('property_type') if hasattr(row, 'get') else None,
                    image_urls=image_urls if image_urls else None,
                    predicted_price=float(row['predicted_price']) if 'predicted_price' in row and pd.notna(row['predicted_price']) else None,
                    price_difference=float(row['price_difference']) if 'price_difference' in row and pd.notna(row['price_difference']) else None,
                    price_difference_percentage=float(row['price_difference_percentage']) if 'price_difference_percentage' in row and pd.notna(row['price_difference_percentage']) else None,
                    bargain_score=float(row['bargain_score']) if 'bargain_score' in row and pd.notna(row['bargain_score']) else None,
                    bargain_category=row.get('bargain_category') if hasattr(row, 'get') else None,
                    is_favorite=False,  # No user authentication in CSV fallback
                    view_count=0
                ))
            
            logger.info(f"Returning {len(csv_properties)} properties from CSV fallback")
            return csv_properties
        
        else:
            logger.info("No properties found in database or CSV")
            return []
            
    except Exception as e:
        logger.error(f"Error searching properties: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/properties/{property_id}", response_model=PropertyResponse)
async def get_property_detail(property_id: int, current_user: Optional[UserInDB] = Depends(get_current_user)):
    """Get detailed information about a specific property"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Only allow access to properties collected by current user (or if no user auth required for viewing)
    if current_user:
        cursor.execute("SELECT * FROM property_listings WHERE id = ? AND collected_by_user_id = ?", (property_id, current_user.id))
    else:
        # If no user authenticated, don't allow access to any properties (user-specific properties only)
        raise HTTPException(status_code=401, detail="Authentication required to view properties")
    
    row = cursor.fetchone()
    
    if row is None:
        raise HTTPException(status_code=404, detail="Property not found or not accessible")
    
    # Update view count
    cursor.execute("""
        UPDATE property_listings SET view_count = view_count + 1 WHERE id = ?
    """, (property_id,))
    conn.commit()
    
    # Get user favorite status
    is_favorite = False
    if current_user:
        fav_cursor = conn.cursor()
        fav_cursor.execute(
            "SELECT property_id FROM user_favorites WHERE user_id = ? AND property_id = ?",
            (current_user.id, property_id)
        )
        is_favorite = fav_cursor.fetchone() is not None
    
    conn.close()
    
    # Handle image_urls - convert from JSON string to list
    image_urls = None
    # Safe access to row columns
    def safe_get_row_value(row, column_name, default=None):
        try:
            return row[column_name] if row[column_name] is not None else default
        except (IndexError, KeyError):
            return default
    
    image_urls_raw = safe_get_row_value(row, 'image_urls')
    if image_urls_raw:
        try:
            if isinstance(image_urls_raw, str):
                # If it's a semicolon-separated string, split it
                if ';' in image_urls_raw:
                    image_urls = [url.strip() for url in image_urls_raw.split(';') if url.strip()]
                else:
                    # Try to parse as JSON
                    image_urls = json.loads(image_urls_raw)
            elif isinstance(image_urls_raw, list):
                image_urls = image_urls_raw
        except:
            image_urls = None
    
    return PropertyResponse(
        id=row['id'],
        title=safe_get_row_value(row, 'title'),
        url=safe_get_row_value(row, 'url'),
        price=row['price'],
        price_per_sqm=safe_get_row_value(row, 'price_per_sqm'),
        area=safe_get_row_value(row, 'area'),
        rooms=safe_get_row_value(row, 'rooms'),
        floor=safe_get_row_value(row, 'floor'),
        total_floors=safe_get_row_value(row, 'total_floors'),
        city=safe_get_row_value(row, 'city'),
        district=safe_get_row_value(row, 'district'),
        address=safe_get_row_value(row, 'address'),
        build_state=safe_get_row_value(row, 'build_state'),
        property_type=safe_get_row_value(row, 'property_type'),
        renovation=safe_get_row_value(row, 'renovation'),
        image_urls=image_urls,
        predicted_price=safe_get_row_value(row, 'predicted_price'),
        price_difference=safe_get_row_value(row, 'price_difference'),
        price_difference_percentage=safe_get_row_value(row, 'price_difference_percentage'),
        bargain_score=safe_get_row_value(row, 'bargain_score'),
        bargain_category=safe_get_row_value(row, 'enhanced_bargain_category', safe_get_row_value(row, 'bargain_category')),
        renovation_category=safe_get_row_value(row, 'renovation_category'),
        global_bargain_category=safe_get_row_value(row, 'global_bargain_category'),
        is_favorite=is_favorite,
        view_count=safe_get_row_value(row, 'view_count', 0),
        # Investment analysis fields
        estimated_monthly_rent=safe_get_row_value(row, 'estimated_monthly_rent'),
        annual_rental_income=safe_get_row_value(row, 'annual_rental_income'),
        gross_rental_yield=safe_get_row_value(row, 'gross_rental_yield'),
        net_rental_yield=safe_get_row_value(row, 'net_rental_yield'),
        roi_percentage=safe_get_row_value(row, 'roi_percentage'),
        payback_period_years=safe_get_row_value(row, 'payback_period_years'),
        monthly_cash_flow=safe_get_row_value(row, 'monthly_cash_flow'),
        investment_category=safe_get_row_value(row, 'investment_category'),
        cash_flow_category=safe_get_row_value(row, 'cash_flow_category'),
        rental_prediction_confidence=safe_get_row_value(row, 'rental_prediction_confidence'),
        # NEW: Renovation cost analysis
        estimated_renovation_cost=safe_get_row_value(row, 'estimated_renovation_cost'),
        renovation_cost_with_buffer=safe_get_row_value(row, 'renovation_cost_with_buffer'),
        total_investment_required=safe_get_row_value(row, 'total_investment_required'),
        renovation_percentage_of_price=safe_get_row_value(row, 'renovation_percentage_of_price'),
        # NEW: Rental premium for renovations
        monthly_rent_premium=safe_get_row_value(row, 'monthly_rent_premium'),
        annual_rent_premium=safe_get_row_value(row, 'annual_rent_premium'),
        renovation_premium_multiplier=safe_get_row_value(row, 'renovation_premium_multiplier'),
        renovation_roi_annual=safe_get_row_value(row, 'renovation_roi_annual'),
        # NEW: Risk assessment
        overall_risk_score=safe_get_row_value(row, 'overall_risk_score'),
        risk_category=safe_get_row_value(row, 'risk_category'),
        renovation_complexity_risk=safe_get_row_value(row, 'renovation_complexity_risk'),
        financial_risk=safe_get_row_value(row, 'financial_risk'),
        market_risk=safe_get_row_value(row, 'market_risk'),
        execution_risk=safe_get_row_value(row, 'execution_risk'),
        # NEW: Final recommendations
        final_investment_recommendation=safe_get_row_value(row, 'final_investment_recommendation'),
        investment_priority_score=safe_get_row_value(row, 'investment_priority_score'),
        investment_priority_category=safe_get_row_value(row, 'investment_priority_category'),
        # NEW: Investment flags
        is_premium_district=safe_get_row_value(row, 'is_premium_district'),
        has_high_renovation_roi=safe_get_row_value(row, 'has_high_renovation_roi'),
        is_fast_payback=safe_get_row_value(row, 'is_fast_payback'),
        has_significant_premium=safe_get_row_value(row, 'has_significant_premium')
    )

# Prediction Routes
@app.post("/properties/predict", response_model=PredictionResponse)
async def predict_rental_price(request: PredictionRequest, current_user: UserInDB = Depends(get_current_active_user)):
    """Predict rental price based on property features using XGBoost rental model"""
    logger.info(f"Received rental prediction request: {request.json()}")
    
    # Feature validation
    if request.area_m2 <= 0:
        raise HTTPException(status_code=400, detail="Area must be greater than 0")
    
    if request.floor is not None and (request.floor < 1 or request.floor > 50):
        raise HTTPException(status_code=400, detail="Floor must be between 1 and 50")
    
    if request.rooms is not None and (request.rooms < 1 or request.rooms > 10):
        raise HTTPException(status_code=400, detail="Rooms must be between 1 and 10")
    
    # Renovation, bathroom, and heating options for rental prediction
    valid_renovation = ["Без ремонта (коробка)", "С ремонтом", "Новый ремонт"]
    if request.renovation and request.renovation not in valid_renovation:
        raise HTTPException(status_code=400, detail="Invalid renovation type")
    
    valid_bathrooms = ["Раздельный", "Совмещенный"]
    if request.bathroom and request.bathroom not in valid_bathrooms:
        raise HTTPException(status_code=400, detail="Invalid bathroom type")
    
    valid_heating = ["Нет", "Есть"]
    if request.heating and request.heating not in valid_heating:
        raise HTTPException(status_code=400, detail="Invalid heating type")
    
    # Load rental model if not already loaded
    if rental_predictor is None:
        load_rental_model()
    
    if rental_predictor is None:
        raise HTTPException(status_code=500, detail="Rental prediction model not available")
    
    # Prepare features for rental prediction
    try:
        property_data = {
            "rooms": request.rooms or 3,
            "area_m2": request.area_m2,
            "floor": request.floor or 3,
            "district": request.district or "Худжанд",  # Default to Khujand since model is trained on Khujand data
            "renovation": request.renovation or "С ремонтом",
            "bathroom": request.bathroom or "Раздельный",
            "heating": request.heating or "Есть"
        }
        
        # Validate and clean input
        validated_data = rental_predictor.validate_input(property_data)
        
        # Make rental prediction
        prediction_result = rental_predictor.predict_rental_price(validated_data)
        
        logger.info(f"Rental prediction successful: {prediction_result['predicted_rental']:.2f} TJS/month")
        
        # Save prediction to user history
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            features_used = {
                "rooms": validated_data.get("rooms"),
                "area_m2": validated_data.get("area_m2"),
                "floor": validated_data.get("floor"),
                "district": validated_data.get("district"),
                "renovation": validated_data.get("renovation"),
                "bathroom": validated_data.get("bathroom"),
                "heating": validated_data.get("heating"),
                "price_per_m2": validated_data.get("price_per_m2")
            }
            
            cursor.execute("""
                INSERT INTO user_predictions (
                    user_id, prediction_input, predicted_price, 
                    confidence_interval, model_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                current_user.id,
                json.dumps(features_used),
                prediction_result['predicted_rental'],
                json.dumps(prediction_result['confidence_interval']),
                'XGBoost Rental',
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Could not save rental prediction to history: {e}")
        
        return PredictionResponse(
            predicted_rental=prediction_result['predicted_rental'],
            confidence_interval_lower=prediction_result['confidence_interval']['lower'],
            confidence_interval_upper=prediction_result['confidence_interval']['upper'],
            annual_rental_income=prediction_result['annual_rental_income'],
            gross_rental_yield=prediction_result['gross_rental_yield'],
            features_used=features_used,
            model_info={}
        )
        
    except Exception as e:
        logger.error(f"Error during rental prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error during rental prediction: {str(e)}")

# Favorite Routes
@app.post("/favorites", response_model=ApiResponse)
async def add_to_favorites(request: FavoriteRequest, current_user: UserInDB = Depends(get_current_active_user)):
    """Add property to favorites"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if already favorited
    cursor.execute(
        "SELECT * FROM user_favorites WHERE user_id = ? AND property_id = ?",
        (current_user.id, request.property_id)
    )
    if cursor.fetchone() is not None:
        raise HTTPException(status_code=400, detail="Property already in favorites")
    
    # Add to favorites
    cursor.execute(
        "INSERT INTO user_favorites (user_id, property_id, notes) VALUES (?, ?, ?)",
        (current_user.id, request.property_id, request.notes)
    )
    conn.commit()
    conn.close()
    
    logger.info(f"Property {request.property_id} added to favorites by user {current_user.id}")
    
    return ApiResponse(
        success=True,
        message="Property added to favorites"
    )

@app.get("/favorites", response_model=List[PropertyResponse])
async def get_favorites(current_user: UserInDB = Depends(get_current_active_user)):
    """Get favorite properties for current user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT p.* FROM property_listings p INNER JOIN user_favorites uf ON p.id = uf.property_id WHERE uf.user_id = ?",
        (current_user.id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    favorites = []
    for row in rows:
        # Safe access to row columns
        def safe_get_row_value(row, column_name, default=None):
            try:
                return row[column_name] if row[column_name] is not None else default
            except (IndexError, KeyError):
                return default
        
        # Handle image_urls - convert from JSON string to list
        image_urls = None
        image_urls_raw = safe_get_row_value(row, 'image_urls')
        if image_urls_raw:
            try:
                if isinstance(image_urls_raw, str):
                    # If it's a semicolon-separated string, split it
                    if ';' in image_urls_raw:
                        image_urls = [url.strip() for url in image_urls_raw.split(';') if url.strip()]
                    else:
                        # Try to parse as JSON
                        image_urls = json.loads(image_urls_raw)
                elif isinstance(image_urls_raw, list):
                    image_urls = image_urls_raw
            except:
                image_urls = None
        
        favorites.append(PropertyResponse(
            id=row['id'],
            title=safe_get_row_value(row, 'title'),
            url=safe_get_row_value(row, 'url'),
            price=row['price'],
            price_per_sqm=safe_get_row_value(row, 'price_per_sqm'),
            area=safe_get_row_value(row, 'area'),
            rooms=safe_get_row_value(row, 'rooms'),
            floor=safe_get_row_value(row, 'floor'),
            total_floors=safe_get_row_value(row, 'total_floors'),
            city=safe_get_row_value(row, 'city'),
            district=safe_get_row_value(row, 'district'),
            address=safe_get_row_value(row, 'address'),
            build_state=safe_get_row_value(row, 'build_state'),
            property_type=safe_get_row_value(row, 'property_type'),
            renovation=safe_get_row_value(row, 'renovation'),
            image_urls=image_urls,
            predicted_price=safe_get_row_value(row, 'predicted_price'),
            price_difference=safe_get_row_value(row, 'price_difference'),
            price_difference_percentage=safe_get_row_value(row, 'price_difference_percentage'),
            bargain_score=safe_get_row_value(row, 'bargain_score'),
            bargain_category=safe_get_row_value(row, 'bargain_category'),
            renovation_category=safe_get_row_value(row, 'renovation_category'),
            global_bargain_category=safe_get_row_value(row, 'global_bargain_category'),
            is_favorite=True,
            view_count=safe_get_row_value(row, 'view_count', 0),
            # Investment analysis fields
            estimated_monthly_rent=safe_get_row_value(row, 'estimated_monthly_rent'),
            annual_rental_income=safe_get_row_value(row, 'annual_rental_income'),
            gross_rental_yield=safe_get_row_value(row, 'gross_rental_yield'),
            net_rental_yield=safe_get_row_value(row, 'net_rental_yield'),
            roi_percentage=safe_get_row_value(row, 'roi_percentage'),
            payback_period_years=safe_get_row_value(row, 'payback_period_years'),
            monthly_cash_flow=safe_get_row_value(row, 'monthly_cash_flow'),
            investment_category=safe_get_row_value(row, 'investment_category'),
            cash_flow_category=safe_get_row_value(row, 'cash_flow_category'),
            rental_prediction_confidence=safe_get_row_value(row, 'rental_prediction_confidence'),
            # NEW: Renovation cost analysis
            estimated_renovation_cost=safe_get_row_value(row, 'estimated_renovation_cost'),
            renovation_cost_with_buffer=safe_get_row_value(row, 'renovation_cost_with_buffer'),
            total_investment_required=safe_get_row_value(row, 'total_investment_required'),
            renovation_percentage_of_price=safe_get_row_value(row, 'renovation_percentage_of_price'),
            # NEW: Rental premium for renovations
            monthly_rent_premium=safe_get_row_value(row, 'monthly_rent_premium'),
            annual_rent_premium=safe_get_row_value(row, 'annual_rent_premium'),
            renovation_premium_multiplier=safe_get_row_value(row, 'renovation_premium_multiplier'),
            renovation_roi_annual=safe_get_row_value(row, 'renovation_roi_annual'),
            # NEW: Risk assessment
            overall_risk_score=safe_get_row_value(row, 'overall_risk_score'),
            risk_category=safe_get_row_value(row, 'risk_category'),
            renovation_complexity_risk=safe_get_row_value(row, 'renovation_complexity_risk'),
            financial_risk=safe_get_row_value(row, 'financial_risk'),
            market_risk=safe_get_row_value(row, 'market_risk'),
            execution_risk=safe_get_row_value(row, 'execution_risk'),
            # NEW: Final recommendations
            final_investment_recommendation=safe_get_row_value(row, 'final_investment_recommendation'),
            investment_priority_score=safe_get_row_value(row, 'investment_priority_score'),
            investment_priority_category=safe_get_row_value(row, 'investment_priority_category'),
            # NEW: Investment flags
            is_premium_district=safe_get_row_value(row, 'is_premium_district'),
            has_high_renovation_roi=safe_get_row_value(row, 'has_high_renovation_roi'),
            is_fast_payback=safe_get_row_value(row, 'is_fast_payback'),
            has_significant_premium=safe_get_row_value(row, 'has_significant_premium')
        ))
    
    return favorites

# Alert Routes
@app.post("/alerts", response_model=ApiResponse)
async def create_alert(alert_data: AlertRequest, current_user: UserInDB = Depends(get_current_active_user)):
    """Create a new alert for properties"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if alert with same name exists
    cursor.execute(
        "SELECT * FROM user_alerts WHERE user_id = ? AND alert_name = ?",
        (current_user.id, alert_data.alert_name)
    )
    if cursor.fetchone() is not None:
        raise HTTPException(status_code=400, detail="Alert with this name already exists")
    
    # Insert new alert
    cursor.execute(
        "INSERT INTO user_alerts (user_id, alert_name, alert_type, conditions) VALUES (?, ?, ?, ?)",
        (current_user.id, alert_data.alert_name, alert_data.alert_type, json.dumps(alert_data.conditions))
    )
    conn.commit()
    conn.close()
    
    logger.info(f"Alert '{alert_data.alert_name}' created for user {current_user.id}")
    
    return ApiResponse(
        success=True,
        message="Alert created successfully"
    )

@app.get("/alerts", response_model=List[AlertRequest])
async def get_alerts(current_user: UserInDB = Depends(get_current_active_user)):
    """Get all alerts for the current user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM user_alerts WHERE user_id = ?",
        (current_user.id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    alerts = []
    for row in rows:
        alerts.append(AlertRequest(
            alert_name=row['alert_name'],
            alert_type=row['alert_type'],
            conditions=json.loads(row['conditions'])
        ))
    
    return alerts

# Statistics Route
@app.get("/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats(current_user: UserInDB = Depends(get_current_active_user)):
    """Get statistics for the dashboard"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get user-specific property statistics
    cursor.execute("SELECT COUNT(*) as total_properties FROM property_listings WHERE collected_by_user_id = ?", (current_user.id,))
    total_properties = cursor.fetchone()['total_properties']
    
    cursor.execute("SELECT COUNT(*) as total_bargains FROM property_listings WHERE collected_by_user_id = ? AND bargain_category IN ('excellent_bargain', 'good_bargain')", (current_user.id,))
    total_bargains = cursor.fetchone()['total_bargains']
    
    cursor.execute("SELECT COUNT(*) as excellent_bargains FROM property_listings WHERE collected_by_user_id = ? AND bargain_category = 'excellent_bargain'", (current_user.id,))
    excellent_bargains = cursor.fetchone()['excellent_bargains']
    
    cursor.execute("SELECT COUNT(*) as good_bargains FROM property_listings WHERE collected_by_user_id = ? AND bargain_category = 'good_bargain'", (current_user.id,))
    good_bargains = cursor.fetchone()['good_bargains']
    
    cursor.execute("SELECT COUNT(*) as user_favorites FROM user_favorites WHERE user_id = ?", (current_user.id,))
    user_favorites = cursor.fetchone()['user_favorites']
    
    cursor.execute("SELECT COUNT(*) as user_searches FROM user_searches WHERE user_id = ?", (current_user.id,))
    user_searches = cursor.fetchone()['user_searches']
    
    cursor.execute("SELECT COUNT(*) as user_predictions FROM user_predictions WHERE user_id = ?", (current_user.id,))
    user_predictions = cursor.fetchone()['user_predictions']
    
    cursor.execute("SELECT COUNT(*) as active_alerts FROM user_alerts WHERE user_id = ? AND active = 1", (current_user.id,))
    active_alerts = cursor.fetchone()['active_alerts']
    
    # Average savings percentage from user's bargains
    cursor.execute("""
        SELECT AVG(price_difference_percentage) as avg_savings_percentage 
        FROM property_listings 
        WHERE collected_by_user_id = ? AND bargain_category IS NOT NULL
    """, (current_user.id,))
    avg_savings_percentage = cursor.fetchone()['avg_savings_percentage'] or 0
    
    conn.close()
    
    return DashboardStats(
        total_properties=total_properties,
        total_bargains=total_bargains,
        excellent_bargains=excellent_bargains,
        good_bargains=good_bargains,
        user_favorites=user_favorites,
        user_searches=user_searches,
        user_predictions=user_predictions,
        active_alerts=active_alerts,
        avg_savings_percentage=avg_savings_percentage
    )

# Data Collection Status Update
def update_user_data_collection_status(user_id: int, has_collected: bool = True):
    """Update user's data collection status"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE users SET has_collected_data = ? WHERE id = ?
    """, (has_collected, user_id))
    
    conn.commit()
    conn.close()
    logger.info(f"Updated data collection status for user {user_id}: {has_collected}")

# Missing API endpoints that frontend is requesting
@app.get("/bargains")
async def get_bargains(
    category: str = Query("all", description="Bargain category: all, excellent, good, fair"),
    limit: int = Query(20, le=100),
    offset: int = Query(0, ge=0),
    current_user: Optional[UserInDB] = Depends(get_current_user_optional)
):
    """Get bargain properties"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build query based on category - only show properties collected by current user
        if current_user:
            query = "SELECT * FROM property_listings WHERE collected_by_user_id = ? AND bargain_category IS NOT NULL"
            params = [current_user.id]
        else:
            # If no user logged in, return empty results (user-specific properties only)
            return []
        
        if category != "all":
            if category == "excellent":
                query += " AND bargain_category = 'exceptional_opportunity'"
            elif category == "good":
                query += " AND bargain_category = 'good_bargain'"
            elif category == "fair":
                query += " AND bargain_category = 'fair_value'"
        
        query += " ORDER BY bargain_score DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Get user favorites if authenticated
        user_favorites = set()
        if current_user:
            fav_cursor = conn.cursor()
            fav_cursor.execute(
                "SELECT property_id FROM user_favorites WHERE user_id = ?",
                (current_user.id,)
            )
            user_favorites = {row[0] for row in fav_cursor.fetchall()}
        
        conn.close()
        
        properties = []
        for row in rows:
            def safe_get_row_value(row, column_name, default=None):
                try:
                    return row[column_name] if row[column_name] is not None else default
                except (IndexError, KeyError):
                    return default
            
            # Handle image_urls
            image_urls = None
            image_urls_raw = safe_get_row_value(row, 'image_urls')
            if image_urls_raw:
                try:
                    if isinstance(image_urls_raw, str):
                        if ';' in image_urls_raw:
                            image_urls = [url.strip() for url in image_urls_raw.split(';') if url.strip()]
                        else:
                            image_urls = json.loads(image_urls_raw)
                    elif isinstance(image_urls_raw, list):
                        image_urls = image_urls_raw
                except:
                    image_urls = None
            
            properties.append(PropertyResponse(
                id=row['id'],
                title=safe_get_row_value(row, 'title'),
                url=safe_get_row_value(row, 'url'),
                price=row['price'],
                price_per_sqm=safe_get_row_value(row, 'price_per_sqm'),
                area=safe_get_row_value(row, 'area'),
                rooms=safe_get_row_value(row, 'rooms'),
                floor=safe_get_row_value(row, 'floor'),
                total_floors=safe_get_row_value(row, 'total_floors'),
                city=safe_get_row_value(row, 'city'),
                district=safe_get_row_value(row, 'district'),
                address=safe_get_row_value(row, 'address'),
                build_state=safe_get_row_value(row, 'build_state'),
                property_type=safe_get_row_value(row, 'property_type'),
                renovation=safe_get_row_value(row, 'renovation'),
                image_urls=image_urls,
                predicted_price=safe_get_row_value(row, 'predicted_price'),
                price_difference=safe_get_row_value(row, 'price_difference'),
                price_difference_percentage=safe_get_row_value(row, 'price_difference_percentage'),
                bargain_score=safe_get_row_value(row, 'bargain_score'),
                bargain_category=safe_get_row_value(row, 'enhanced_bargain_category') or safe_get_row_value(row, 'bargain_category'),  # Prefer enhanced category
                renovation_category=safe_get_row_value(row, 'renovation_category'),
                global_bargain_category=safe_get_row_value(row, 'global_bargain_category'),
                is_favorite=row['id'] in user_favorites,
                view_count=safe_get_row_value(row, 'view_count', 0),
                # Investment analysis fields
                estimated_monthly_rent=safe_get_row_value(row, 'estimated_monthly_rent'),
                annual_rental_income=safe_get_row_value(row, 'annual_rental_income'),
                gross_rental_yield=safe_get_row_value(row, 'gross_rental_yield'),
                net_rental_yield=safe_get_row_value(row, 'net_rental_yield'),
                roi_percentage=safe_get_row_value(row, 'roi_percentage'),
                payback_period_years=safe_get_row_value(row, 'payback_period_years'),
                monthly_cash_flow=safe_get_row_value(row, 'monthly_cash_flow'),
                investment_category=safe_get_row_value(row, 'investment_category'),
                cash_flow_category=safe_get_row_value(row, 'cash_flow_category'),
                rental_prediction_confidence=safe_get_row_value(row, 'rental_prediction_confidence'),
                # NEW: Renovation cost analysis
                estimated_renovation_cost=safe_get_row_value(row, 'estimated_renovation_cost'),
                renovation_cost_with_buffer=safe_get_row_value(row, 'renovation_cost_with_buffer'),
                total_investment_required=safe_get_row_value(row, 'total_investment_required'),
                renovation_percentage_of_price=safe_get_row_value(row, 'renovation_percentage_of_price'),
                # NEW: Rental premium for renovations
                monthly_rent_premium=safe_get_row_value(row, 'monthly_rent_premium'),
                annual_rent_premium=safe_get_row_value(row, 'annual_rent_premium'),
                renovation_premium_multiplier=safe_get_row_value(row, 'renovation_premium_multiplier'),
                renovation_roi_annual=safe_get_row_value(row, 'renovation_roi_annual'),
                # NEW: Risk assessment
                overall_risk_score=safe_get_row_value(row, 'overall_risk_score'),
                risk_category=safe_get_row_value(row, 'risk_category'),
                renovation_complexity_risk=safe_get_row_value(row, 'renovation_complexity_risk'),
                financial_risk=safe_get_row_value(row, 'financial_risk'),
                market_risk=safe_get_row_value(row, 'market_risk'),
                execution_risk=safe_get_row_value(row, 'execution_risk'),
                # NEW: Final recommendations
                final_investment_recommendation=safe_get_row_value(row, 'final_investment_recommendation'),
                investment_priority_score=safe_get_row_value(row, 'investment_priority_score'),
                investment_priority_category=safe_get_row_value(row, 'investment_priority_category'),
                # NEW: Investment flags
                is_premium_district=safe_get_row_value(row, 'is_premium_district'),
                has_high_renovation_roi=safe_get_row_value(row, 'has_high_renovation_roi'),
                is_fast_payback=safe_get_row_value(row, 'is_fast_payback'),
                has_significant_premium=safe_get_row_value(row, 'has_significant_premium')
            ))
        
        return properties
        
    except Exception as e:
        logger.error(f"Error fetching bargains: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch bargains: {str(e)}")

@app.get("/districts")
async def get_districts():
    """Get list of available districts"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT district 
            FROM property_listings 
            WHERE district IS NOT NULL AND district != '' 
            ORDER BY district
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        districts = [row[0] for row in rows if row[0]]
        
        return {"districts": districts}
        
    except Exception as e:
        logger.error(f"Error fetching districts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch districts: {str(e)}")

@app.get("/filter-ranges", response_model=FilterRanges)
async def get_filter_ranges():
    """Get filter ranges for price, area, and floor"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get price ranges
        cursor.execute("""
            SELECT MIN(price), MAX(price) 
            FROM property_listings 
            WHERE price IS NOT NULL AND price > 0
        """)
        price_result = cursor.fetchone()
        
        # Get area ranges
        cursor.execute("""
            SELECT MIN(area), MAX(area) 
            FROM property_listings 
            WHERE area IS NOT NULL AND area > 0
        """)
        area_result = cursor.fetchone()
        
        # Get floor ranges
        cursor.execute("""
            SELECT MIN(floor), MAX(floor) 
            FROM property_listings 
            WHERE floor IS NOT NULL AND floor > 0
        """)
        floor_result = cursor.fetchone()
        
        conn.close()
        
        return FilterRanges(
            price_min=float(price_result[0]) if price_result[0] else 0.0,
            price_max=float(price_result[1]) if price_result[1] else 1000000.0,
            area_min=float(area_result[0]) if area_result[0] else 0.0,
            area_max=float(area_result[1]) if area_result[1] else 500.0,
            floor_min=float(floor_result[0]) if floor_result[0] else 1.0,
            floor_max=float(floor_result[1]) if floor_result[1] else 50.0
        )
        
    except Exception as e:
        logger.error(f"Error fetching filter ranges: {e}")
        # Return default ranges in case of error
        return FilterRanges(
            price_min=0.0,
            price_max=1000000.0,
            area_min=0.0,
            area_max=500.0,
            floor_min=1.0,
            floor_max=50.0
        )

@app.get("/market-stats", response_model=MarketStatsResponse)
async def get_market_stats():
    """Get comprehensive market statistics for analytics dashboard"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Total listings
        cursor.execute("SELECT COUNT(*) FROM property_listings")
        total_listings = cursor.fetchone()[0]
        
        # Price statistics
        cursor.execute("""
            SELECT AVG(price), MIN(price), MAX(price) 
            FROM property_listings 
            WHERE price IS NOT NULL AND price > 0
        """)
        price_stats = cursor.fetchone()
        avg_price = float(price_stats[0]) if price_stats[0] else 0
        min_price = float(price_stats[1]) if price_stats[1] else 0
        max_price = float(price_stats[2]) if price_stats[2] else 0
        
        # Median price
        cursor.execute("""
            SELECT price FROM property_listings 
            WHERE price IS NOT NULL AND price > 0 
            ORDER BY price
        """)
        prices = [row[0] for row in cursor.fetchall()]
        median_price = float(prices[len(prices)//2]) if prices else 0
        
        # Average price per sqm
        cursor.execute("""
            SELECT AVG(price_per_sqm) 
            FROM property_listings 
            WHERE price_per_sqm IS NOT NULL AND price_per_sqm > 0
        """)
        avg_price_per_sqm_result = cursor.fetchone()
        avg_price_per_sqm = float(avg_price_per_sqm_result[0]) if avg_price_per_sqm_result[0] else 0
        
        # Bargain statistics - only count actual investment opportunities
        cursor.execute("""
            SELECT COUNT(*) FROM property_listings 
            WHERE bargain_category IN ('exceptional_opportunity', 'good_bargain')
        """)
        total_bargains = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM property_listings 
            WHERE bargain_category = 'exceptional_opportunity'
        """)
        excellent_bargains = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM property_listings 
            WHERE bargain_category = 'good_bargain'
        """)
        good_bargains = cursor.fetchone()[0]
        
        # Price distribution by ranges - updated to match actual data range (min: 220k TJS)
        price_ranges = {
            "Under 400k": 0,
            "400k-500k": 0,
            "500k-600k": 0,
            "600k-700k": 0,
            "700k-800k": 0,
            "800k-1M": 0,
            "1M-1.5M": 0,
            "1.5M-2M": 0,
            "2M+": 0
        }
        
        cursor.execute("SELECT price FROM property_listings WHERE price IS NOT NULL")
        for row in cursor.fetchall():
            price = row[0]
            if price < 400000:
                price_ranges["Under 400k"] += 1
            elif price < 500000:
                price_ranges["400k-500k"] += 1
            elif price < 600000:
                price_ranges["500k-600k"] += 1
            elif price < 700000:
                price_ranges["600k-700k"] += 1
            elif price < 800000:
                price_ranges["700k-800k"] += 1
            elif price < 1000000:
                price_ranges["800k-1M"] += 1
            elif price < 1500000:
                price_ranges["1M-1.5M"] += 1
            elif price < 2000000:
                price_ranges["1.5M-2M"] += 1
            else:
                price_ranges["2M+"] += 1
        
        # Room distribution
        cursor.execute("""
            SELECT rooms, COUNT(*) 
            FROM property_listings 
            WHERE rooms IS NOT NULL 
            GROUP BY rooms 
            ORDER BY rooms
        """)
        room_distribution = {str(row[0]): row[1] for row in cursor.fetchall()}
        
        # City distribution
        cursor.execute("""
            SELECT city, COUNT(*) 
            FROM property_listings 
            WHERE city IS NOT NULL AND city != '' 
            GROUP BY city 
            ORDER BY COUNT(*) DESC
        """)
        city_distribution = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return MarketStatsResponse(
            total_listings=total_listings,
            avg_price=avg_price,
            median_price=median_price,
            min_price=min_price,
            max_price=max_price,
            avg_price_per_sqm=avg_price_per_sqm,
            total_bargains=total_bargains,
            excellent_bargains=excellent_bargains,
            good_bargains=good_bargains,
            price_distribution=price_ranges,
            room_distribution=room_distribution,
            city_distribution=city_distribution
        )
        
    except Exception as e:
        logger.error(f"Error fetching market stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch market statistics: {str(e)}")

@app.get("/district-investment-scores")
async def get_district_investment_scores():
    """Get district-wise investment scores for analytics dashboard"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get district investment scores
        cursor.execute("""
            SELECT district, AVG(investment_score), COUNT(*) 
            FROM property_listings 
            WHERE district IS NOT NULL 
            AND district != '' 
            AND investment_score IS NOT NULL 
            GROUP BY district 
            HAVING COUNT(*) >= 5
            ORDER BY AVG(investment_score) DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            # Return empty data if no investment scores available
            return {
                "labels": [],
                "values": [],
                "counts": []
            }
        
        labels = []
        values = []
        counts = []
        
        for row in rows:
            labels.append(row[0])
            values.append(float(row[1]))
            counts.append(int(row[2]))
        
        return {
            "labels": labels,
            "values": values,
            "counts": counts
        }
        
    except Exception as e:
        logger.error(f"Error fetching district investment scores: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch district investment scores: {str(e)}")

# Chart Data Endpoints for Market Analytics Dashboard
@app.get("/chart-data/renovation-impact")
async def get_renovation_impact_data():
    """Get renovation impact data for charts"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT property_type, AVG(price), COUNT(*) 
            FROM property_listings 
            WHERE property_type IS NOT NULL 
            GROUP BY property_type 
            ORDER BY AVG(price) DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {
                "labels": ["Без ремонта (коробка)", "Новый ремонт", "С ремонтом"],
                "values": [5351, 8420, 7755],
                "counts": [198, 130, 111]
            }
        
        labels = []
        values = []
        counts = []
        
        for row in rows:
            labels.append(row[0])
            values.append(float(row[1]))
            counts.append(int(row[2]))
        
        return {
            "labels": labels,
            "values": values,
            "counts": counts
        }
        
    except Exception as e:
        logger.error(f"Error fetching renovation impact data: {e}")
        # Return fallback data
        return {
            "labels": ["Без ремонта (коробка)", "Новый ремонт", "С ремонтом"],
            "values": [5351, 8420, 7755],
            "counts": [198, 130, 111]
        }

@app.get("/chart-data/build-type-analysis")
async def get_build_type_analysis_data():
    """Get build type analysis data for charts"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT build_state, AVG(price), COUNT(*) 
            FROM property_listings 
            WHERE build_state IS NOT NULL 
            GROUP BY build_state 
            ORDER BY AVG(price) DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {
                "labels": ["Вторичный рынок", "Новостройка"],
                "values": [7780, 6516],
                "counts": [122, 317]
            }
        
        labels = []
        values = []
        counts = []
        
        for row in rows:
            labels.append(row[0])
            values.append(float(row[1]))
            counts.append(int(row[2]))
        
        return {
            "labels": labels,
            "values": values,
            "counts": counts
        }
        
    except Exception as e:
        logger.error(f"Error fetching build type analysis data: {e}")
        # Return fallback data
        return {
            "labels": ["Вторичный рынок", "Новостройка"],
            "values": [7780, 6516],
            "counts": [122, 317]
        }

@app.get("/chart-data/market-segments")
async def get_market_segments_data():
    """Get market segments data for charts"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Define price segments
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN price < 400000 THEN 'Budget Market'
                    WHEN price < 600000 THEN 'Mid Market'
                    WHEN price < 800000 THEN 'Premium Market'
                    ELSE 'Luxury Market'
                END as segment,
                AVG(price), COUNT(*)
            FROM property_listings 
            WHERE price IS NOT NULL AND price > 0
            GROUP BY segment
            ORDER BY AVG(price)
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {
                "labels": ["Budget Market", "Mid Market", "Premium Market", "Luxury Market"],
                "values": [517120, 544223, 587870, 882175],
                "counts": [108, 230, 23, 55]
            }
        
        labels = []
        values = []
        counts = []
        
        for row in rows:
            labels.append(row[0])
            values.append(float(row[1]))

            counts.append(int(row[2]))
        
        return {
            "labels": labels,
            "values": values,
            "counts": counts
        }
        
    except Exception as e:
        logger.error(f"Error fetching market segments data: {e}")
        # Return fallback data
        return {
            "labels": ["Budget Market", "Mid Market", "Premium Market", "Luxury Market"],
            "values": [517120, 544223, 587870, 882175],
            "counts": [108, 230, 23, 55]
        }

@app.get("/chart-data/size-analysis")
async def get_size_analysis_data():
    """Get size analysis data for charts"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Define size segments
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN area < 60 THEN 'Compact (53m²)'
                    WHEN area < 75 THEN 'Standard (68m²)'
                    WHEN area < 90 THEN 'Spacious (83m²)'
                    ELSE 'Premium (106m²)'
                END as size_segment,
                AVG(price), COUNT(*)
            FROM property_listings 
            WHERE area IS NOT NULL AND area > 0
            GROUP BY size_segment
            ORDER BY AVG(price)
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {
                "labels": ["Compact (53m²)", "Standard (68m²)", "Spacious (83m²)", "Premium (106m²)"],
                "values": [8466, 8074, 6458, 6364],
                "counts": [17, 100, 153, 169]
            }
        
        labels = []
        values = []
        counts = []
        
        for row in rows:
            labels.append(row[0])
            values.append(float(row[1]))
            counts.append(int(row[2]))
        
        return {
            "labels": labels,
            "values": values,
            "counts": counts
        }
        
    except Exception as e:
        logger.error(f"Error fetching size analysis data: {e}")
        # Return fallback data
        return {
            "labels": ["Compact (53m²)", "Standard (68m²)", "Spacious (83m²)", "Premium (106m²)"],
            "values": [8466, 8074, 6458, 6364],
            "counts": [17, 100, 153, 169]
        }

@app.get("/chart-data/bargain-distribution")
async def get_bargain_distribution_data():
    """Get bargain distribution data for charts"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT bargain_category, COUNT(*) 
            FROM property_listings 
            WHERE bargain_category IS NOT NULL 
            GROUP BY bargain_category 
            ORDER BY COUNT(*) DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {
                "labels": ["Excellent Deals", "Good Investments", "Fair Value", "Market Price", "Overpriced"],
                "values": [2, 23, 185, 141, 88],
                "counts": [2, 23, 185, 141, 88]
            }
        
        # Map database bargain categories to chart labels
        category_mapping = {
            "excellent": "Excellent Deals",
            "good": "Good Investments", 
            "fair": "Fair Value",
            "market": "Market Price",
            "overpriced": "Overpriced"
        }
        
        labels = []
        values = []
        counts = []
        
        for row in rows:
            category = row[0]
            count = int(row[1])
            
            label = category_mapping.get(category, category)
            labels.append(label)
            values.append(count)  # For bargain distribution, values and counts are the same
            counts.append(count)
        
        return {
            "labels": labels,
            "values": values,
            "counts": counts
        }
        
    except Exception as e:
        logger.error(f"Error fetching bargain distribution data: {e}")
        # Return fallback data
        return {
            "labels": ["Excellent Deals", "Good Investments", "Fair Value", "Market Price", "Overpriced"],
            "values": [2, 23, 185, 141, 88],
            "counts": [2, 23, 185, 141, 88]
        }

@app.get("/chart-data/activity-by-day")
async def get_activity_by_day_data():
    """Get activity by day data for charts"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Use publication_weekday data for activity distribution
        cursor.execute("""
            SELECT 
                publication_weekday as day_name,
                COUNT(*) as activity_count
            FROM property_listings 
            WHERE publication_weekday IS NOT NULL
            GROUP BY publication_weekday
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        # Initialize with all days in proper order
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        activity_data = {day: 0 for day in days}
        
        # Fill in actual data
        for row in rows:
            if row[0]:  # Check if day_name is not None
                activity_data[row[0]] = int(row[1])
        
        # If no data, return mock data
        if all(value == 0 for value in activity_data.values()):
            return {
                "x": days,
                "y": [45, 65, 72, 85, 78, 68, 60]  # Mock weekly pattern
            }
        
        return {
            "x": days,
            "y": [activity_data[day] for day in days]
        }
        
    except Exception as e:
        logger.error(f"Error fetching activity by day data: {e}")
        # Return fallback data
        return {
            "x": ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'],
            "y": [45, 65, 72, 85, 78, 68, 60]
        }

# Data Collection Routes

# Test pipeline endpoint removed - use full pipeline instead

@app.get("/data/pipeline-status")
async def get_pipeline_status():
    """Get basic pipeline status"""
    try:
        # Check if data directories exist
        somon_project_dir = Path(__file__).parent.parent.parent
        data_dirs = {
            "raw": somon_project_dir / "data" / "raw",
            "preprocessed": somon_project_dir / "data" / "preprocessed", 
            "feature_engineered": somon_project_dir / "data" / "feature_engineered"
        }
        
        status = {}
        for name, path in data_dirs.items():
            if path.exists():
                files = list(path.glob("*.csv"))
                status[name] = {
                    "exists": True,
                    "file_count": len(files),
                    "latest_file": files[-1].name if files else None
                }
            else:
                status[name] = {"exists": False, "file_count": 0, "latest_file": None}
        
        # Check database status
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM property_listings")
        db_count = cursor.fetchone()[0]
        conn.close()
        
        return {
            "database_properties": db_count,
            "data_directories": status,
            "pipeline_ready": True
        }
        
    except Exception as e:
        logger.error(f"Error checking pipeline status: {e}")
        return {
            "database_properties": 0,
            "data_directories": {},
            "pipeline_ready": False,
            "error": str(e)
        }

@app.post("/data/scrape", response_model=ApiResponse)
async def run_scraping(
    background_tasks: BackgroundTasks,
    scraping_params: ScrapingRequest,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Run data scraping with specified parameters"""
    try:
        logger.info(f"Starting scraping for user {current_user.id} with params: {scraping_params}")
        
        # Initialize tracker
        tracker = DataCollectionTracker()
        start_time = time.time()
        
        # Create output directory and file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        somon_project_dir = Path(__file__).parent.parent.parent
        output_dir = somon_project_dir / "data" / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use relative path for scrapy (since we change to somon_project_dir)
        relative_output_file = f"data/raw/scraped_listings_{timestamp}.csv"
        absolute_output_file = output_dir / f"scraped_listings_{timestamp}.csv"
        
        # Change to project root for scrapy
        import os
        original_cwd = os.getcwd()
        os.chdir(somon_project_dir)
        
        # Build scrapy command - use relative path
        cmd = [
            "scrapy", "crawl", "somon_spider",
            "-a", f"rooms={scraping_params.rooms}",
            "-a", f"city={scraping_params.city}",
            "-a", f"build_state={scraping_params.build_state}",
            "-o", relative_output_file,
            "-s", "FEED_FORMAT=csv",
            "-s", "DOWNLOAD_DELAY=0.1",  # Much faster - 0.1 second delay
            "-s", "RANDOMIZE_DOWNLOAD_DELAY=False",  # Disable randomization for speed
            "-s", "CONCURRENT_REQUESTS=16",  # Increase concurrent requests
            "-s", "CONCURRENT_REQUESTS_PER_DOMAIN=8",  # More requests per domain
            "-s", "AUTOTHROTTLE_ENABLED=True",  # Auto-throttle for optimal speed
            "-s", "AUTOTHROTTLE_START_DELAY=0.1",  # Start with fast requests
            "-s", "AUTOTHROTTLE_MAX_DELAY=1",  # Max 1 second delay
            "-s", "AUTOTHROTTLE_TARGET_CONCURRENCY=8.0",  # Target 8 concurrent requests
            "-s", "RETRY_ENABLED=True",  # Enable retries for failed requests
            "-s", "RETRY_TIMES=3",  # Retry failed requests up to 3 times
            "-s", "COOKIES_ENABLED=False",  # Disable cookies for speed
            "-L", "INFO"
        ]
        
        if scraping_params.property_type and scraping_params.property_type.strip():
            cmd.extend(["-a", f"property_type={scraping_params.property_type}"])
        
        # Run scrapy
        logger.info(f"🚀 Starting FULL UNLIMITED scraping pipeline...")
        logger.info(f"📋 Parameters: {scraping_params.dict()}")
        logger.info(f"⚡ Optimized for speed with concurrent requests and no pagination limits")
        logger.info(f"💾 Output file: {relative_output_file}")
        logger.info(f"🔧 Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 60 min timeout for large datasets
        os.chdir(original_cwd)
        
        duration = time.time() - start_time
        
        logger.info(f"Scrapy completed with return code: {result.returncode}")
        logger.info(f"Scrapy stdout (last 500 chars): {result.stdout[-500:] if result.stdout else 'No stdout'}")
        logger.info(f"Scrapy stderr (last 500 chars): {result.stderr[-500:] if result.stderr else 'No stderr'}")
        
        # Check results using absolute path
        records_scraped = 0
        if absolute_output_file.exists() and absolute_output_file.stat().st_size > 0:
            try:
                # Use more robust CSV reading with error handling for malformed rows
                try:
                    # Try with multiple encoding options and error handling strategies
                    for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
                        try:
                            df = pd.read_csv(absolute_output_file, encoding=encoding, on_bad_lines='skip')
                            records_scraped = len(df)
                            logger.info(f"Successfully read {records_scraped} records from {absolute_output_file} with encoding {encoding}")
                            break
                        except UnicodeDecodeError:
                            continue
                        except Exception as e:
                            if "36 values for 21 columns" in str(e):
                                logger.warning(f"Detected the mysterious pandas error: {e}")
                                # Try alternative reading method
                                import csv
                                with open(absolute_output_file, 'r', encoding=encoding) as f:
                                    reader = csv.reader(f)
                                    rows = list(reader)
                                    records_scraped = len(rows) - 1  # Subtract header
                                logger.info(f"Alternative CSV reading successful: {records_scraped} records")
                                break
                            else:
                                raise
                    else:
                        raise Exception("All encoding attempts failed")
                        
                except Exception as pandas_error:
                    logger.warning(f"Pandas failed to read CSV: {pandas_error}")
                    
                    # Fallback: manual CSV parsing with error handling
                    try:
                        valid_rows = 0
                        with open(absolute_output_file, 'r', encoding='utf-8') as f:
                            # Skip header
                            next(f)
                            for line_num, line in enumerate(f, 2):
                                # Count fields in line
                                fields = line.strip().split(',')
                                if len(fields) == 15:  # Expected number of columns
                                    valid_rows += 1
                        
                        records_scraped = valid_rows
                        logger.info(f"Manual count found {records_scraped} valid records (skipped malformed rows)")
                    except Exception as manual_error:
                        logger.warning(f"Manual CSV parsing also failed: {manual_error}")
                        # Just count lines as fallback
                        with open(absolute_output_file, 'r') as f:
                            records_scraped = sum(1 for line in f) - 1  # Subtract header
                        logger.info(f"Fallback line count: {records_scraped} records")
                        
            except Exception as e:
                logger.warning(f"Could not read scraped CSV: {e}")
                import traceback
                logger.warning(f"CSV reading traceback: {traceback.format_exc()}")
        else:
            logger.warning(f"Output file does not exist or is empty: {absolute_output_file}")
        
        # Log the scraping stage
        tracker.log_pipeline_stage(
            stage="scraping",
            output_file=str(absolute_output_file),
            records_processed=records_scraped,
            parameters=scraping_params.dict(),
            status="completed" if result.returncode == 0 else "failed",
            duration=duration,
            error_log=result.stderr if result.returncode != 0 else None
        )
        
        # Update user data collection status
        if records_scraped > 0:
            update_user_data_collection_status(current_user.id, True)
        
        return ApiResponse(
            success=result.returncode == 0 and records_scraped > 0,
            message=f"🎉 FULL scraping completed! {records_scraped} properties extracted with NO pagination limits.",
            data={
                "records_scraped": records_scraped,
                "output_file": str(absolute_output_file),
                "duration_seconds": duration,
                "scrapy_return_code": result.returncode,
                "scrapy_stdout": result.stdout[-1000:] if result.stdout else "",  # Last 1000 chars
                "scrapy_stderr": result.stderr[-1000:] if result.stderr else "",  # Last 1000 chars
                "scrapy_command": " ".join(cmd)  # The actual command that was run
            }
        )
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Scraping error: {e}")
        logger.error(f"Full traceback: {error_trace}")
        return ApiResponse(
            success=False,
            message=f"Scraping failed: {str(e)}",
            data={"error": str(e), "traceback": error_trace}
        )

@app.post("/data/preprocess", response_model=ApiResponse)
async def run_preprocessing(
    background_tasks: BackgroundTasks,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Run data preprocessing on the latest raw data"""
    try:
        logger.info(f"Starting preprocessing for user {current_user.id}")
        
        # Initialize tracker and file manager
        tracker = DataCollectionTracker()
        file_manager = SmartFileManager(tracker)
        
        # Get latest raw data file
        somon_project_dir = Path(__file__).parent.parent.parent
        raw_dir = somon_project_dir / "data" / "raw"
        latest_raw_file = file_manager.get_latest_file(raw_dir, "*.csv")
        
        if not latest_raw_file:
            return ApiResponse(
                success=False,
                message="No raw data files found for preprocessing",
                data={}
            )
        
        # Create output directory
        preprocessed_dir = somon_project_dir / "data" / "preprocessed"
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        # Change to project root
        import os
        original_cwd = os.getcwd()
        os.chdir(somon_project_dir)
        
        # Run preprocessing script - use -o flag for output directory
        cmd = [
            "python", "utils/preprocess_listings_v2.py",
            str(latest_raw_file),
            "-o", str(preprocessed_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        os.chdir(original_cwd)
        
        duration = time.time() - start_time
        
        # Check results - find the generated output file
        records_processed = 0
        output_file = None
        
        # Look for the generated cleaned_listings_v2.csv file
        potential_files = list(preprocessed_dir.glob("cleaned_listings_v2*.csv"))
        if potential_files:
            # Get the most recent file
            output_file = max(potential_files, key=lambda x: x.stat().st_mtime)
            try:
                df = pd.read_csv(output_file)
                records_processed = len(df)
                logger.info(f"Found preprocessed file: {output_file} with {records_processed} records")
            except Exception as e:
                logger.warning(f"Could not read preprocessed CSV: {e}")
        else:
            logger.warning("No preprocessed output file found")
        
        # Log the preprocessing stage
        tracker.log_pipeline_stage(
            stage="preprocessing",
            source_file=str(latest_raw_file),
            output_file=str(output_file),
            records_processed=records_processed,
            status="completed" if result.returncode == 0 else "failed",
            duration=duration,
            error_log=result.stderr if result.returncode != 0 else None
        )
        
        return ApiResponse(
            success=result.returncode == 0 and records_processed > 0,
            message=f"Preprocessing completed. {records_processed} records processed.",
            data={
                "input_file": str(latest_raw_file),
                "output_file": str(output_file),
                "records_processed": records_processed,
                "duration_seconds": duration
            }
        )
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return ApiResponse(
            success=False,
            message=f"Preprocessing failed: {str(e)}",
            data={"error": str(e)}
        )

@app.post("/data/feature-engineering", response_model=ApiResponse)
async def run_feature_engineering(
    background_tasks: BackgroundTasks,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Run feature engineering on the latest preprocessed data"""
    try:
        logger.info(f"Starting feature engineering for user {current_user.id}")
        
        # Initialize tracker and file manager
        tracker = DataCollectionTracker()
        file_manager = SmartFileManager(tracker)
        
        # Get latest preprocessed data file
        somon_project_dir = Path(__file__).parent.parent.parent
        preprocessed_dir = somon_project_dir / "data" / "preprocessed"
        latest_preprocessed_file = file_manager.get_latest_file(preprocessed_dir, "*.csv")
        
        if not latest_preprocessed_file:
            return ApiResponse(
                success=False,
                message="No preprocessed data files found for feature engineering",
                data={}
            )
        
        # Create output directory
        feature_dir = somon_project_dir / "data" / "feature_engineered"
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        # Change to project root
        import os
        original_cwd = os.getcwd()
        os.chdir(somon_project_dir)
        
        # Run feature engineering script - use -o flag for output directory
        cmd = [
            "python", "utils/feature_engineering_enhanced.py",
            str(latest_preprocessed_file),
            "-o", str(feature_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        os.chdir(original_cwd)
        
        duration = time.time() - start_time
        
        # Check results - find the generated output file
        records_processed = 0
        output_file = None
        
        # Look for the generated listings_with_features.csv file
        potential_files = list(feature_dir.glob("listings_with_features*.csv"))
        if potential_files:
            # Get the most recent file
            output_file = max(potential_files, key=lambda x: x.stat().st_mtime)
            try:
                df = pd.read_csv(output_file)
                records_processed = len(df)
                logger.info(f"Found feature-engineered file: {output_file} with {records_processed} records")
            except Exception as e:
                logger.warning(f"Could not read feature-engineered CSV: {e}")
        else:
            logger.warning("No feature-engineered output file found")
        
        # Log the feature engineering stage
        tracker.log_pipeline_stage(
            stage="feature_engineering",
            source_file=str(latest_preprocessed_file),
            output_file=str(output_file),
            records_processed=records_processed,
            status="completed" if result.returncode == 0 else "failed",
            duration=duration,
            error_log=result.stderr if result.returncode != 0 else None
        )
        
        return ApiResponse(
            success=result.returncode == 0 and records_processed > 0,
            message=f"Feature engineering completed. {records_processed} records processed.",
            data={
                "input_file": str(latest_preprocessed_file),
                "output_file": str(output_file),
                "records_processed": records_processed,
                "duration_seconds": duration
            }
        )
        
    except Exception as e:
        logger.error(f"Feature engineering error: {e}")
        return ApiResponse(
            success=False,
            message=f"Feature engineering failed: {str(e)}",
            data={"error": str(e)}
        )

@app.post("/data/import-to-database", response_model=ApiResponse)
async def import_to_database(
    background_tasks: BackgroundTasks,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Import the latest feature-engineered data to database"""
    try:
        logger.info(f"Starting database import for user {current_user.id}")
        
        # Initialize tracker and file manager
        tracker = DataCollectionTracker()
        file_manager = SmartFileManager(tracker)
        
        # Get latest feature-engineered data file
        somon_project_dir = Path(__file__).parent.parent.parent
        feature_dir = somon_project_dir / "data" / "feature_engineered"
        latest_feature_file = file_manager.get_latest_file(feature_dir, "*.csv")
        
        logger.info(f"Looking for files in: {feature_dir}")
        logger.info(f"Latest feature file found: {latest_feature_file}")
        
        if not latest_feature_file:
            return ApiResponse(
                success=False,
                message="No feature-engineered data files found for database import",
                data={}
            )
        
        start_time = time.time()
        
        # Load CSV data
        df = pd.read_csv(latest_feature_file)
        records_to_import = len(df)
        
        # Debug: Log column names and sample data
        logger.info(f"CSV columns: {list(df.columns)}")
        logger.info(f"Sample row keys: {df.iloc[0].to_dict().keys() if len(df) > 0 else 'No data'}")
        if len(df) > 0:
            logger.info(f"Sample values: price={df.iloc[0].get('price', 'N/A')}, area_m2={df.iloc[0].get('area_m2', 'N/A')}, url={df.iloc[0].get('url', 'N/A')}")
        
        # Import to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Clear existing data for this user only (user-specific property isolation)
        logger.info(f"Clearing previous properties for user {current_user.id} to show only latest search results")
        cursor.execute("DELETE FROM property_listings WHERE collected_by_user_id = ?", (current_user.id,))
        cursor.execute("DELETE FROM user_favorites WHERE user_id = ? AND property_id IN (SELECT id FROM property_listings WHERE collected_by_user_id = ?)", (current_user.id, current_user.id))
        logger.info(f"Previous properties for user {current_user.id} cleared, importing new data")
        
        imported_count = 0
        failed_count = 0
        for _, row in df.iterrows():
            try:
                # Handle image URLs
                image_urls = None
                if 'image_urls' in row and pd.notna(row['image_urls']):
                    if isinstance(row['image_urls'], str):
                        image_urls = row['image_urls']
                    else:
                        image_urls = str(row['image_urls'])
                
                # Debug: Log the first row to see what data we're trying to import
                if imported_count == 0:
                    logger.info(f"Sample row data: URL={row.get('url')}, price={row.get('price')}, area={row.get('area_m2')}")
                
                cursor.execute("""
                    INSERT OR REPLACE INTO property_listings (
                        collected_by_user_id, url, price, area, rooms, floor, total_floors, city, district, 
                        build_state, property_type, price_per_sqm, predicted_price,
                        price_difference, price_difference_percentage, bargain_score,
                        bargain_category, investment_score, image_urls, photo_count,
                        publication_weekday, created_at, updated_at,
                        estimated_monthly_rent, annual_rental_income, gross_rental_yield,
                        net_rental_yield, roi_percentage, payback_period_years, monthly_cash_flow,
                        investment_category, cash_flow_category, rental_prediction_confidence,
                        enhanced_bargain_score, enhanced_bargain_category, investment_score_v2,
                        risk_adjusted_investment_score,
                        -- Renovation columns
                        renovation, base_renovation_cost, estimated_renovation_cost, renovation_cost_with_buffer,
                        total_investment_required, renovation_percentage_of_price,
                        monthly_rent_premium, annual_rent_premium, renovation_premium_multiplier,
                        renovation_roi_annual, renovation_impact_on_yield, renovation_payback_years,
                        overall_risk_score, risk_category, renovation_complexity_risk,
                        financial_risk, market_risk, execution_risk,
                        preliminary_investment_recommendation, final_investment_recommendation,
                        investment_priority_score, investment_priority_category,
                        is_premium_district, has_high_renovation_roi, is_fast_payback, has_significant_premium,
                        renovation_score,
                        -- Missing columns with default values
                        title, address, price_to_area_ratio, floor_ratio, is_ground_floor, is_top_floor, 
                        is_middle_floor, area_category, room_density, district_avg_price, district_price_ratio,
                        district_avg_area, district_area_ratio, city_avg_price, city_price_ratio, 
                        city_avg_area, city_area_ratio, scraped_date, is_active, first_seen, last_seen, view_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    current_user.id,  # collected_by_user_id
                    row.get('url'),
                    row.get('price'),
                    row.get('area_m2'),
                    row.get('rooms'),
                    row.get('floor'),
                    None,  # total_floors not in CSV
                    None,  # city not in CSV
                    row.get('district'),
                    row.get('build_type'),  # maps to build_state
                    row.get('property_type'),  # Keep property_type separate from renovation
                    row.get('price_per_sqm'),
                    None,  # predicted_price not in CSV
                    None,  # price_difference not in CSV
                    None,  # price_difference_percentage not in CSV
                    row.get('bargain_score'),
                    row.get('bargain_category'),
                    row.get('value_score'),  # maps to investment_score
                    image_urls,
                    row.get('photo_count'),
                    row.get('publication_weekday'),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    # New investment columns
                    row.get('estimated_monthly_rent'),
                    row.get('annual_rental_income'),
                    row.get('gross_rental_yield'),
                    row.get('net_rental_yield'),
                    row.get('roi_percentage'),
                    row.get('payback_period_years'),
                    row.get('monthly_cash_flow'),
                    row.get('investment_category'),
                    row.get('cash_flow_category'),
                    row.get('rental_prediction_confidence'),
                    row.get('enhanced_bargain_score'),
                    row.get('enhanced_bargain_category'),
                    row.get('investment_score'),  # maps to investment_score_v2
                    row.get('risk_adjusted_investment_score'),
                    # Renovation columns
                    row.get('renovation'),
                    row.get('base_renovation_cost'),
                    row.get('estimated_renovation_cost'),
                    row.get('renovation_cost_with_buffer'),
                    row.get('total_investment_required'),
                    row.get('renovation_percentage_of_price'),
                    row.get('monthly_rent_premium'),
                    row.get('annual_rent_premium'),
                    row.get('renovation_premium_multiplier'),
                    row.get('renovation_roi_annual'),
                    row.get('renovation_impact_on_yield'),
                    row.get('renovation_payback_years'),
                    row.get('overall_risk_score'),
                    row.get('risk_category'),
                    row.get('renovation_complexity_risk'),
                    row.get('financial_risk'),
                    row.get('market_risk'),
                    row.get('execution_risk'),
                    row.get('preliminary_investment_recommendation'),
                    row.get('final_investment_recommendation'),
                    row.get('investment_priority_score'),
                    row.get('investment_priority_category'),
                    row.get('is_premium_district'),
                    row.get('has_high_renovation_roi'),
                    row.get('is_fast_payback'),
                    row.get('has_significant_premium'),
                    row.get('renovation_score'),
                    # Missing columns with default values
                    None,  # title (not available in CSV)
                    None,  # address (not available in CSV)
                    row.get('price_to_area_ratio'),
                    row.get('floor_ratio'),
                    row.get('is_ground_floor'),
                    row.get('is_top_floor'),
                    row.get('is_middle_floor'),
                    row.get('area_category'),
                    row.get('room_density'),
                    row.get('district_avg_price'),
                    row.get('district_price_ratio'),
                    row.get('district_avg_area'),
                    row.get('district_area_ratio'),
                    row.get('city_avg_price'),
                    row.get('city_price_ratio'),
                    row.get('city_avg_area'),
                    row.get('city_area_ratio'),
                    datetime.now().isoformat(),  # scraped_date
                    True,  # is_active
                    datetime.now().isoformat(),  # first_seen
                    datetime.now().isoformat(),  # last_seen
                    0  # view_count
                ))
                imported_count += 1
                
                # Log progress every 10 records
                if imported_count % 10 == 0 or imported_count <= 5:
                    logger.info(f"Imported {imported_count} records so far...")
                    
            except Exception as e:
                logger.warning(f"Error importing row {failed_count + imported_count + 1}: {e}")
                logger.warning(f"Failed row data: {dict(row)}")
                failed_count += 1
                continue
        
        logger.info(f"Database import completed: {imported_count} successful, {failed_count} failed")
        
        conn.commit()
        conn.close()
        
        duration = time.time() - start_time
        
        # Log the database import stage
        tracker.log_pipeline_stage(
            stage="database_import",
            source_file=str(latest_feature_file),
            records_processed=records_to_import,
            records_imported=imported_count,
            status="completed",
            duration=duration
        )
        
        # Reload properties data
        load_properties_data()
        
        return ApiResponse(
            success=True,
            message=f"✅ Latest search imported! {imported_count} new properties now available (previous data cleared).",
            data={
                "input_file": str(latest_feature_file),
                "records_imported": imported_count,
                "records_total": records_to_import,
                "duration_seconds": duration,
                "database_cleared": True,
                "showing_latest_search_only": True
            }
        )
        
    except Exception as e:
        logger.error(f"Database import error: {e}")
        return ApiResponse(
            success=False,
            message=f"Database import failed: {str(e)}",
            data={"error": str(e)}
        )

@app.post("/data/run-full-pipeline", response_model=ApiResponse)
async def run_full_pipeline(
    background_tasks: BackgroundTasks,
    scraping_params: ScrapingRequest,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Run the complete data collection pipeline"""
    try:
        logger.info(f"Starting full pipeline for user {current_user.id}")
        
        results = {
            "scraping": None,
            "preprocessing": None,
            "feature_engineering": None,
            "database_import": None
        }
        
        # Step 1: Scraping
        logger.info("Pipeline Step 1: Scraping")
        scrape_result = await run_scraping(background_tasks, scraping_params, current_user)
        results["scraping"] = scrape_result.dict()
        
        if not scrape_result.success:
            return ApiResponse(
                success=False,
                message="Pipeline failed at scraping stage",
                data=results
            )
        
        # Step 2: Preprocessing
        logger.info("Pipeline Step 2: Preprocessing")
        preprocess_result = await run_preprocessing(background_tasks, current_user)
        results["preprocessing"] = preprocess_result.dict()
        
        if not preprocess_result.success:
            return ApiResponse(
                success=False,
                message="Pipeline failed at preprocessing stage",
                data=results
            )
        
        # Step 3: Feature Engineering
        logger.info("Pipeline Step 3: Feature Engineering")
        feature_result = await run_feature_engineering(background_tasks, current_user)
        results["feature_engineering"] = feature_result.dict()
        
        if not feature_result.success:
            return ApiResponse(
                success=False,
                message="Pipeline failed at feature engineering stage",
                data=results
            )
        
        # Step 4: Database Import
        logger.info("Pipeline Step 4: Database Import")
        import_result = await import_to_database(background_tasks, current_user)
        results["database_import"] = import_result.dict()
        
        if not import_result.success:
            return ApiResponse(
                success=False,
                message="Pipeline failed at database import stage",
                data=results
            )
        
        # Update user data collection status
        update_user_data_collection_status(current_user.id, True)
        
        # Get the final count from database import results
        import_data = results.get("database_import", {}).get("data", {})
        imported_count = import_data.get("records_imported", 0)
        
        return ApiResponse(
            success=True,
            message=f"🎉 PIPELINE COMPLETE! {imported_count} fresh properties ready to view (previous search cleared).",
            data=results
        )
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Full pipeline error: {e}")
        logger.error(f"Full pipeline traceback: {error_trace}")
        return ApiResponse(
            success=False,
            message=f"Full pipeline failed: {str(e)}",
            data={"error": str(e), "traceback": error_trace, "results": results}
        )

# User History Routes
@app.get("/user/searches")
async def get_user_search_history(current_user: UserInDB = Depends(get_current_active_user)):
    """Get user's search history"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, search_type, search_params, results_count, created_at
            FROM user_searches 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT 50
        """, (current_user.id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        searches = []
        for row in rows:
            searches.append({
                "id": row[0],
                "search_type": row[1],
                "search_params": json.loads(row[2]) if row[2] else {},
                "results_count": row[3],
                "created_at": row[4]
            })
        
        return searches
        
    except Exception as e:
        logger.error(f"Error fetching user search history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch search history")

@app.get("/user/predictions")
async def get_user_prediction_history(current_user: UserInDB = Depends(get_current_active_user)):
    """Get user's prediction history"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, prediction_input, predicted_price, confidence_interval, created_at
            FROM user_predictions 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT 50
        """, (current_user.id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        predictions = []
        for row in rows:
            predictions.append({
                "id": row[0],
                "property_features": json.loads(row[1]) if row[1] else {},
                "predicted_price": row[2],
                "confidence_interval": json.loads(row[3]) if row[3] else {},
                "created_at": row[4]
            })
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error fetching user prediction history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch prediction history")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

# New Investment Analytics Endpoints

@app.get("/analytics/roi-distribution")
async def get_roi_distribution():
    """Get ROI distribution for investment analysis."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN roi_percentage >= 10 THEN 'Excellent (10%+)'
                    WHEN roi_percentage >= 7 THEN 'Good (7-10%)'
                    WHEN roi_percentage >= 5 THEN 'Fair (5-7%)'
                    WHEN roi_percentage >= 3 THEN 'Poor (3-5%)'
                    ELSE 'Avoid (<3%)'
                END as roi_category,
                COUNT(*) as count,
                AVG(roi_percentage) as avg_roi
            FROM property_listings 
            WHERE roi_percentage IS NOT NULL
            GROUP BY roi_category
            ORDER BY avg_roi DESC
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        return {
            "labels": [row[0] for row in results],
            "values": [row[2] for row in results],  # avg_roi
            "counts": [row[1] for row in results]   # count
        }
    except Exception as e:
        logger.error(f"Error getting ROI distribution: {e}")
        return {"labels": [], "values": [], "counts": []}

@app.get("/analytics/payback-analysis")
async def get_payback_analysis():
    """Get payback period analysis for investment planning."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN payback_period_years <= 10 THEN 'Quick (≤10y)'
                    WHEN payback_period_years <= 15 THEN 'Moderate (10-15y)'
                    WHEN payback_period_years <= 20 THEN 'Slow (15-20y)'
                    WHEN payback_period_years <= 30 THEN 'Very Slow (20-30y)'
                    ELSE 'Poor (30y+)'
                END as payback_category,
                COUNT(*) as count,
                AVG(payback_period_years) as avg_payback
            FROM property_listings 
            WHERE payback_period_years IS NOT NULL AND payback_period_years < 50
            GROUP BY payback_category
            ORDER BY avg_payback ASC
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        return {
            "labels": [row[0] for row in results],
            "values": [row[2] for row in results],  # avg_payback
            "counts": [row[1] for row in results]   # count
        }
    except Exception as e:
        logger.error(f"Error getting payback analysis: {e}")
        return {"labels": [], "values": [], "counts": []}

@app.get("/analytics/rental-yield-by-district")
async def get_rental_yield_by_district():
    """Get rental yield analysis by district."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                district,
                COUNT(*) as count,
                AVG(net_rental_yield) as avg_yield,
                AVG(estimated_monthly_rent) as avg_rent,
                AVG(price) as avg_price
            FROM property_listings 
            WHERE net_rental_yield IS NOT NULL AND district IS NOT NULL
            GROUP BY district
            HAVING COUNT(*) >= 3
            ORDER BY avg_yield DESC
            LIMIT 10
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        return {
            "labels": [row[0] for row in results],
            "values": [row[2] for row in results],  # avg_yield
            "counts": [row[1] for row in results],  # count
            "avg_rent": [row[3] for row in results],  # avg_rent
            "avg_price": [row[4] for row in results]  # avg_price
        }
    except Exception as e:
        logger.error(f"Error getting rental yield by district: {e}")
        return {"labels": [], "values": [], "counts": [], "avg_rent": [], "avg_price": []}

@app.get("/analytics/investment-vs-price")
async def get_investment_vs_price():
    """Get investment score vs price analysis for scatter plot."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                price,
                roi_percentage,
                net_rental_yield,
                payback_period_years,
                district,
                bargain_category,
                investment_category
            FROM property_listings 
            WHERE roi_percentage IS NOT NULL 
                AND price IS NOT NULL 
                AND payback_period_years < 50
            ORDER BY roi_percentage DESC
            LIMIT 200
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        return {
            "price": [row[0] for row in results],
            "roi": [row[1] for row in results],
            "yield": [row[2] for row in results],
            "payback": [row[3] for row in results],
            "district": [row[4] for row in results],
            "bargain_category": [row[5] for row in results],
            "investment_category": [row[6] for row in results]
        }
    except Exception as e:
        logger.error(f"Error getting investment vs price data: {e}")
        return {"price": [], "roi": [], "yield": [], "payback": [], "district": [], "bargain_category": [], "investment_category": []}

@app.get("/analytics/cash-flow-distribution")
async def get_cash_flow_distribution():
    """Get monthly cash flow distribution."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN monthly_cash_flow >= 500 THEN 'Excellent (500+ TJS)'
                    WHEN monthly_cash_flow >= 300 THEN 'Good (300-500 TJS)'
                    WHEN monthly_cash_flow >= 100 THEN 'Moderate (100-300 TJS)'
                    WHEN monthly_cash_flow >= 0 THEN 'Break Even (0-100 TJS)'
                    ELSE 'Negative (<0 TJS)'
                END as cash_flow_category,
                COUNT(*) as count,
                AVG(monthly_cash_flow) as avg_cash_flow
            FROM property_listings 
            WHERE monthly_cash_flow IS NOT NULL
            GROUP BY cash_flow_category
            ORDER BY avg_cash_flow DESC
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        return {
            "labels": [row[0] for row in results],
            "values": [row[2] for row in results],  # avg_cash_flow
            "counts": [row[1] for row in results]   # count
        }
    except Exception as e:
        logger.error(f"Error getting cash flow distribution: {e}")
        return {"labels": [], "values": [], "counts": []}


# Test endpoint for welcome email
@app.post("/test/welcome-email")
async def test_welcome_email(email: str, name: str = "Test User"):
    """Test endpoint to send welcome email"""
    try:
        logger.info(f"🧪 Testing welcome email for {email}")
        
        # Import with proper path
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        from services.notification_service import EmailNotificationService
        
        email_service = EmailNotificationService()
        logger.info(f"📧 Email service configured: {email_service.is_configured}")
        
        if not email_service.is_configured:
            return {"success": False, "message": "Email service not configured", "configured": False}
        
        success = email_service.send_welcome_email(email, name)
        
        return {
            "success": success,
            "message": "Welcome email sent successfully" if success else "Failed to send welcome email",
            "configured": email_service.is_configured,
            "email": email,
            "name": name
        }
        
    except Exception as e:
        logger.error(f"❌ Error in test welcome email: {str(e)}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "message": f"Error: {str(e)}",
            "configured": False
        }


# Production deployment configuration
if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()
    
    # Use environment variables for production
    host = args.host
    port = int(os.getenv("PORT", args.port))
    
    uvicorn.run(app, host=host, port=port)
