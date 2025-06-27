#!/usr/bin/env python3
"""
Initialize database tables for the Real Estate Platform

This script creates all necessary database tables including:
- Users table for authentication
- Property listings table with all features
- User interaction tables (favorites, searches, etc.)
"""

import sqlite3
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_database_tables():
    """Create all necessary database tables"""
    db_path = "real_estate.db"
    
    # Create database connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    logger.info("Creating database tables...")
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email VARCHAR(255) UNIQUE NOT NULL,
            username VARCHAR(100) UNIQUE,
            full_name VARCHAR(255),
            hashed_password VARCHAR(255) NOT NULL,
            is_active BOOLEAN DEFAULT 1,
            is_verified BOOLEAN DEFAULT 0,
            has_collected_data BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            email_notifications BOOLEAN DEFAULT 1,
            push_notifications BOOLEAN DEFAULT 1,
            notification_frequency VARCHAR(50) DEFAULT 'daily',
            notification_enabled BOOLEAN DEFAULT 1
        )
    """)
    
    # Property listings table with all necessary columns
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS property_listings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_by_user_id INTEGER NOT NULL,
            title VARCHAR(500),
            url VARCHAR(1000) NOT NULL,
            price REAL NOT NULL,
            price_per_sqm REAL,
            area REAL,
            rooms INTEGER,
            floor INTEGER,
            total_floors INTEGER,
            city VARCHAR(100),
            district VARCHAR(100),
            address VARCHAR(500),
            build_state VARCHAR(50),
            property_type VARCHAR(50),
            image_urls TEXT,
            photo_count INTEGER DEFAULT 0,
            publication_weekday VARCHAR(20),
            
            -- Feature engineering columns
            price_to_area_ratio REAL,
            floor_ratio REAL,
            is_ground_floor BOOLEAN,
            is_top_floor BOOLEAN,
            is_middle_floor BOOLEAN,
            area_category VARCHAR(20),
            room_density REAL,
            district_avg_price REAL,
            district_price_ratio REAL,
            district_avg_area REAL,
            district_area_ratio REAL,
            city_avg_price REAL,
            city_price_ratio REAL,
            city_avg_area REAL,
            city_area_ratio REAL,
            
            -- ML and bargain detection columns
            predicted_price REAL,
            price_difference REAL,
            price_difference_percentage REAL,
            investment_score REAL,
            bargain_score REAL,
            bargain_category VARCHAR(20),
            
            -- Investment analysis columns
            estimated_monthly_rent REAL,
            annual_rental_income REAL,
            gross_rental_yield REAL,
            net_rental_yield REAL,
            roi_percentage REAL,
            payback_period_years REAL,
            monthly_cash_flow REAL,
            investment_category VARCHAR(50),
            cash_flow_category VARCHAR(50),
            rental_prediction_confidence REAL,
            enhanced_bargain_score REAL,
            enhanced_bargain_category VARCHAR(50),
            investment_score_v2 REAL,
            risk_adjusted_investment_score REAL,
            
            -- Renovation analysis columns
            renovation TEXT,
            base_renovation_cost REAL,
            estimated_renovation_cost REAL,
            renovation_cost_with_buffer REAL,
            total_investment_required REAL,
            renovation_percentage_of_price REAL,
            
            -- Rental premium for renovations
            monthly_rent_premium REAL,
            annual_rent_premium REAL,
            renovation_premium_multiplier REAL,
            renovation_roi_annual REAL,
            renovation_impact_on_yield REAL,
            renovation_payback_years REAL,
            
            -- Risk assessment
            overall_risk_score REAL,
            risk_category TEXT,
            renovation_complexity_risk REAL,
            financial_risk REAL,
            market_risk REAL,
            execution_risk REAL,
            
            -- Final recommendations
            preliminary_investment_recommendation TEXT,
            final_investment_recommendation TEXT,
            investment_priority_score REAL,
            investment_priority_category TEXT,
            
            -- Investment flags
            is_premium_district BOOLEAN,
            has_high_renovation_roi BOOLEAN,
            is_fast_payback BOOLEAN,
            has_significant_premium BOOLEAN,
            
            -- Additional scoring fields
            renovation_score INTEGER,
            
            -- Metadata columns
            scraped_date TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            view_count INTEGER DEFAULT 0,
            
            FOREIGN KEY (collected_by_user_id) REFERENCES users (id) ON DELETE CASCADE,
            UNIQUE(url, collected_by_user_id)
        )
    """)
    
    # User favorites table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            property_id INTEGER NOT NULL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
            FOREIGN KEY (property_id) REFERENCES property_listings (id) ON DELETE CASCADE,
            UNIQUE(user_id, property_id)
        )
    """)
    
    # User searches table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            search_type VARCHAR(50) NOT NULL,
            search_params TEXT,
            results_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)
    
    # User predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            prediction_input TEXT,
            predicted_price REAL,
            confidence_interval TEXT,
            model_version VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)
    
    # User alerts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            alert_name VARCHAR(255) NOT NULL,
            alert_type VARCHAR(50) NOT NULL,
            conditions TEXT,
            active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_triggered TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
            UNIQUE(user_id, alert_name)
        )
    """)
    
    # Data collection history table - Enhanced version tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_collection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            pipeline_version VARCHAR(50),
            pipeline_stage VARCHAR(50) NOT NULL,
            source_file_path TEXT,
            source_file_name VARCHAR(255),
            source_file_size INTEGER,
            source_file_hash VARCHAR(64),
            output_file_path TEXT,
            output_file_name VARCHAR(255),
            output_file_size INTEGER,
            records_processed INTEGER,
            records_imported INTEGER,
            updated_properties INTEGER,
            processing_status VARCHAR(50) DEFAULT 'completed',
            processing_duration_seconds REAL,
            scraping_parameters TEXT,
            preprocessing_parameters TEXT,
            feature_engineering_version VARCHAR(50),
            feature_engineering_parameters TEXT,
            bargain_detection_stats TEXT,
            data_quality_metrics TEXT,
            error_log TEXT,
            created_by VARCHAR(100),
            metadata TEXT,
            is_active BOOLEAN DEFAULT 1
        )
    """)
    
    # Create indexes for better performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_listings_collected_by_user ON property_listings (collected_by_user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_listings_price ON property_listings (price)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_listings_city ON property_listings (city)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_listings_district ON property_listings (district)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_listings_bargain ON property_listings (bargain_category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_listings_renovation ON property_listings (renovation)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_listings_risk_category ON property_listings (risk_category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_listings_final_recommendation ON property_listings (final_investment_recommendation)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_listings_renovation_score ON property_listings (renovation_score)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_favorites_user ON user_favorites (user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_searches_user ON user_searches (user_id)")
    
    # Commit changes
    conn.commit()
    conn.close()
    
    logger.info("Database tables created successfully!")
    return True

if __name__ == "__main__":
    logger.info("Initializing Real Estate Platform Database...")
    success = create_database_tables()
    if success:
        logger.info("Database initialization completed successfully!")
    else:
        logger.error("Database initialization failed!")
