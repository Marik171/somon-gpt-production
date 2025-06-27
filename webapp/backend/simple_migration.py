#!/usr/bin/env python3
"""
Simple Database Migration Script for User-Specific Properties

This script adds the collected_by_user_id column and assigns existing properties to the first user.
"""

import sqlite3
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_migrate():
    """Simple migration to add collected_by_user_id column"""
    db_path = "real_estate.db"
    
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
        
        logger.info("Starting simple database migration...")
        
        # Get the first user ID
        cursor.execute("SELECT id FROM users ORDER BY id LIMIT 1")
        first_user = cursor.fetchone()
        
        if not first_user:
            logger.error("No users found in database - cannot migrate")
            return False
        
        first_user_id = first_user[0]
        logger.info(f"Will assign all existing properties to user ID {first_user_id}")
        
        # Add the column with a default value
        cursor.execute(f"ALTER TABLE property_listings ADD COLUMN collected_by_user_id INTEGER DEFAULT {first_user_id}")
        logger.info("Added collected_by_user_id column")
        
        # Update all existing properties to belong to the first user
        cursor.execute("UPDATE property_listings SET collected_by_user_id = ? WHERE collected_by_user_id IS NULL", (first_user_id,))
        updated_count = cursor.rowcount
        logger.info(f"Updated {updated_count} existing properties to belong to user {first_user_id}")
        
        # Create index for the new column
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_listings_collected_by_user ON property_listings (collected_by_user_id)")
        logger.info("Created index for collected_by_user_id")
        
        # Commit changes
        conn.commit()
        logger.info("Simple migration completed successfully!")
        
        # Verify migration
        cursor.execute("SELECT COUNT(*) FROM property_listings WHERE collected_by_user_id = ?", (first_user_id,))
        user_properties = cursor.fetchone()[0]
        logger.info(f"Verification: {user_properties} properties now belong to user {first_user_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    logger.info("Starting simple database migration...")
    success = simple_migrate()
    if success:
        logger.info("Migration completed successfully!")
    else:
        logger.error("Migration failed!") 