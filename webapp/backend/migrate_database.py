#!/usr/bin/env python3
"""
Database Migration Script for User-Specific Properties

This script migrates the existing database to support user-specific properties by:
1. Adding collected_by_user_id column to property_listings table
2. Creating a default user for existing properties
3. Updating the table structure with foreign key constraints
"""

import sqlite3
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_database():
    """Migrate database to support user-specific properties"""
    db_path = "real_estate.db"
    
    # Check if database exists
    if not Path(db_path).exists():
        logger.info("Database doesn't exist yet, no migration needed")
        return True
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if migration is already done
        cursor.execute("PRAGMA table_info(property_listings)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'collected_by_user_id' in columns:
            logger.info("Migration already completed - collected_by_user_id column exists")
            return True
        
        logger.info("Starting database migration for user-specific properties...")
        
        # Step 1: Check if there are any existing properties
        cursor.execute("SELECT COUNT(*) FROM property_listings")
        existing_properties_count = cursor.fetchone()[0]
        logger.info(f"Found {existing_properties_count} existing properties")
        
        # Step 2: Create a migration user if there are existing properties
        migration_user_id = None
        if existing_properties_count > 0:
            # Check if migration user already exists
            cursor.execute("SELECT id FROM users WHERE email = 'migration@system.local'")
            existing_migration_user = cursor.fetchone()
            
            if existing_migration_user:
                migration_user_id = existing_migration_user[0]
                logger.info(f"Using existing migration user with ID {migration_user_id}")
            else:
                # Create migration user for existing properties
                cursor.execute("""
                    INSERT INTO users (
                        email, username, full_name, hashed_password, 
                        is_active, has_collected_data, created_at,
                        email_notifications, push_notifications, notification_frequency
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    'migration@system.local',
                    'migration_user',
                    'Migration User (Legacy Properties)',
                    '$2b$12$dummy.hash.for.migration.user.placeholder',  # Dummy hash, account disabled
                    0,  # Inactive account
                    1,  # Has collected data
                    '2024-01-01 00:00:00',
                    0,  # No email notifications
                    0,  # No push notifications
                    'never'
                ))
                migration_user_id = cursor.lastrowid
                logger.info(f"Created migration user with ID {migration_user_id}")
        
        # Step 3: Add the new column with a default value
        if existing_properties_count > 0 and migration_user_id:
            # Add column with default value for existing properties
            cursor.execute(f"ALTER TABLE property_listings ADD COLUMN collected_by_user_id INTEGER DEFAULT {migration_user_id}")
            logger.info("Added collected_by_user_id column with default migration user")
            
            # Update all existing properties to belong to migration user
            cursor.execute("UPDATE property_listings SET collected_by_user_id = ? WHERE collected_by_user_id IS NULL", (migration_user_id,))
            updated_count = cursor.rowcount
            logger.info(f"Updated {updated_count} existing properties to belong to migration user")
        else:
            # No existing properties, just add the column without default
            cursor.execute("ALTER TABLE property_listings ADD COLUMN collected_by_user_id INTEGER")
            logger.info("Added collected_by_user_id column (no existing properties)")
        
        # Step 4: Create new property_listings table with proper constraints
        logger.info("Creating new property_listings table with foreign key constraints...")
        
        # Rename existing table
        cursor.execute("ALTER TABLE property_listings RENAME TO property_listings_old")
        
        # Create new table with proper structure
        cursor.execute("""
            CREATE TABLE property_listings (
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
        
        # Step 5: Copy data from old table to new table
        # Check if collected_by_user_id already exists in old table
        cursor.execute("PRAGMA table_info(property_listings_old)")
        old_columns = [column[1] for column in cursor.fetchall()]
        
        if 'collected_by_user_id' in old_columns:
            # Column already exists, do a simple copy
            cursor.execute("INSERT INTO property_listings SELECT * FROM property_listings_old")
        else:
            # Need to add the collected_by_user_id value during copy
            other_columns = [col for col in old_columns if col != 'collected_by_user_id']
            columns_str = ', '.join(['collected_by_user_id'] + other_columns)
            values_str = ', '.join([str(migration_user_id)] + other_columns)
            
            copy_query = f"""
                INSERT INTO property_listings ({columns_str})
                SELECT {values_str} FROM property_listings_old
            """
            cursor.execute(copy_query)
        copied_count = cursor.rowcount
        logger.info(f"Copied {copied_count} properties to new table structure")
        
        # Step 6: Drop old table
        cursor.execute("DROP TABLE property_listings_old")
        logger.info("Dropped old property_listings table")
        
        # Step 7: Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_listings_collected_by_user ON property_listings (collected_by_user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_listings_price ON property_listings (price)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_listings_city ON property_listings (city)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_listings_district ON property_listings (district)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_listings_bargain ON property_listings (bargain_category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_listings_renovation ON property_listings (renovation)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_listings_risk_category ON property_listings (risk_category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_listings_final_recommendation ON property_listings (final_investment_recommendation)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_listings_renovation_score ON property_listings (renovation_score)")
        logger.info("Created database indexes")
        
        # Commit all changes
        conn.commit()
        logger.info("Database migration completed successfully!")
        
        # Step 8: Verify migration
        cursor.execute("SELECT COUNT(*) FROM property_listings")
        final_count = cursor.fetchone()[0]
        logger.info(f"Verification: {final_count} properties in migrated database")
        
        if migration_user_id:
            cursor.execute("SELECT COUNT(*) FROM property_listings WHERE collected_by_user_id = ?", (migration_user_id,))
            migration_user_properties = cursor.fetchone()[0]
            logger.info(f"Verification: {migration_user_properties} properties belong to migration user")
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    logger.info("Starting database migration for user-specific properties...")
    success = migrate_database()
    if success:
        logger.info("Migration completed successfully!")
    else:
        logger.error("Migration failed!") 