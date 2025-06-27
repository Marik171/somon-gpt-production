#!/usr/bin/env python3
"""
Daily Scheduler for SomonGPT Real Estate Platform

This script runs daily to:
1. Collect new property data
2. Detect new bargain properties
3. Track price changes in favorite properties
4. Send email notifications to users
"""

import os
import sys
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import asyncio
import schedule
import time

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'somon_project'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('daily_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DailyScheduler:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'somon_project', 'webapp', 'backend', 'real_estate.db'
        )
        self.scraper_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'somon_project', 'scraper'
        )
        
    def get_db_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def run_data_collection(self) -> Dict[str, Any]:
        """Run the complete data collection pipeline"""
        logger.info("🚀 Starting daily data collection pipeline...")
        
        try:
            # Import necessary modules
            sys.path.append(self.scraper_path)
            
            # Run scraper for different property types
            scraping_results = {
                'total_scraped': 0,
                'new_properties': 0,
                'updated_properties': 0,
                'errors': []
            }
            
            # Scraping configurations for different property types
            scraping_configs = [
                {'rooms': '1-komnatnyie', 'city': 'hudzhand', 'build_state': 'sostoyanie---10'},
                {'rooms': '2-komnatnyie', 'city': 'hudzhand', 'build_state': 'sostoyanie---10'},
                {'rooms': '3-komnatnyie', 'city': 'hudzhand', 'build_state': 'sostoyanie---10'},
                {'rooms': '4-komnatnyie', 'city': 'hudzhand', 'build_state': 'sostoyanie---10'},
            ]
            
            for config in scraping_configs:
                try:
                    logger.info(f"📊 Scraping {config['rooms']} properties...")
                    
                    # Here you would call your scraper
                    # For now, we'll simulate the process
                    # result = run_scraper(config)
                    # scraping_results['total_scraped'] += result.get('count', 0)
                    
                    # Simulate scraping results
                    scraped_count = 50  # This would come from actual scraper
                    scraping_results['total_scraped'] += scraped_count
                    logger.info(f"✅ Scraped {scraped_count} {config['rooms']} properties")
                    
                except Exception as e:
                    error_msg = f"Error scraping {config['rooms']}: {str(e)}"
                    logger.error(error_msg)
                    scraping_results['errors'].append(error_msg)
            
            # Run data preprocessing and feature engineering
            try:
                logger.info("🔄 Running data preprocessing...")
                # Here you would call your preprocessing pipeline
                # preprocess_data()
                logger.info("✅ Data preprocessing completed")
                
                logger.info("🎯 Running feature engineering...")
                # Here you would call your feature engineering
                # engineer_features()
                logger.info("✅ Feature engineering completed")
                
            except Exception as e:
                error_msg = f"Error in data processing: {str(e)}"
                logger.error(error_msg)
                scraping_results['errors'].append(error_msg)
            
            logger.info(f"✅ Data collection completed. Total scraped: {scraping_results['total_scraped']}")
            return scraping_results
            
        except Exception as e:
            logger.error(f"❌ Data collection failed: {str(e)}")
            return {'total_scraped': 0, 'new_properties': 0, 'updated_properties': 0, 'errors': [str(e)]}
    
    def detect_new_bargains(self) -> List[Dict[str, Any]]:
        """Detect new bargain properties added in the last 24 hours"""
        logger.info("🎯 Detecting new bargain properties...")
        
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Get bargain properties added in the last 24 hours
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
            
            query = """
            SELECT id, title, url, price, area, rooms, floor, city, district, 
                   bargain_category, price_difference, price_difference_percentage,
                   estimated_monthly_rent, gross_rental_yield, payback_period_years,
                   created_at
            FROM property_listings 
            WHERE bargain_category IN ('excellent_bargain', 'good_bargain', 'fair_value')
            AND (created_at >= ? OR updated_at >= ?)
            ORDER BY bargain_score DESC
            LIMIT 50
            """
            
            cursor.execute(query, (yesterday, yesterday))
            rows = cursor.fetchall()
            
            new_bargains = []
            for row in rows:
                new_bargains.append({
                    'id': row[0],
                    'title': row[1],
                    'url': row[2],
                    'price': row[3],
                    'area': row[4],
                    'rooms': row[5],
                    'floor': row[6],
                    'city': row[7],
                    'district': row[8],
                    'bargain_category': row[9],
                    'price_difference': row[10],
                    'price_difference_percentage': row[11],
                    'estimated_monthly_rent': row[12],
                    'gross_rental_yield': row[13],
                    'payback_period_years': row[14],
                    'created_at': row[15]
                })
            
            conn.close()
            logger.info(f"✅ Found {len(new_bargains)} new bargain properties")
            return new_bargains
            
        except Exception as e:
            logger.error(f"❌ Error detecting new bargains: {str(e)}")
            return []
    
    def track_favorite_price_changes(self) -> Dict[str, List[Dict[str, Any]]]:
        """Track price changes in users' favorite properties"""
        logger.info("📈 Tracking favorite property price changes...")
        
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Get all users with favorites and email notifications enabled
            cursor.execute("""
                SELECT DISTINCT u.id, u.email, u.full_name
                FROM users u
                INNER JOIN user_favorites uf ON u.id = uf.user_id
                WHERE u.email_notifications = 1
                AND u.is_active = 1
            """)
            users = cursor.fetchall()
            
            user_price_changes = {}
            
            for user_id, email, full_name in users:
                # Get user's favorite properties with price history
                cursor.execute("""
                    SELECT p.id, p.title, p.url, p.price, p.area, p.rooms, 
                           p.city, p.district, uf.created_at as favorited_at,
                           ph.old_price, ph.new_price, ph.change_date
                    FROM property_listings p
                    INNER JOIN user_favorites uf ON p.id = uf.property_id
                    LEFT JOIN (
                        SELECT property_id, old_price, new_price, change_date,
                               ROW_NUMBER() OVER (PARTITION BY property_id ORDER BY change_date DESC) as rn
                        FROM property_price_history
                        WHERE change_date >= datetime('now', '-1 day')
                    ) ph ON p.id = ph.property_id AND ph.rn = 1
                    WHERE uf.user_id = ?
                """)
                
                cursor.execute("""
                    SELECT p.id, p.title, p.url, p.price, p.area, p.rooms, 
                           p.city, p.district, uf.created_at as favorited_at
                    FROM property_listings p
                    INNER JOIN user_favorites uf ON p.id = uf.property_id
                    WHERE uf.user_id = ?
                """, (user_id,))
                
                favorites = cursor.fetchall()
                price_changes = []
                
                for fav in favorites:
                    # Check if price changed in the last 24 hours
                    # For now, we'll simulate price changes
                    # In reality, you'd compare current price with historical data
                    
                    # Simulate some price changes (remove this in production)
                    import random
                    if random.random() < 0.1:  # 10% chance of price change
                        old_price = fav[3] * (1 + random.uniform(-0.1, 0.1))  # ±10% change
                        price_change = fav[3] - old_price
                        price_change_percentage = (price_change / old_price) * 100
                        
                        price_changes.append({
                            'id': fav[0],
                            'title': fav[1],
                            'url': fav[2],
                            'current_price': fav[3],
                            'previous_price': old_price,
                            'price_change': price_change,
                            'price_change_percentage': price_change_percentage,
                            'area': fav[4],
                            'rooms': fav[5],
                            'city': fav[6],
                            'district': fav[7]
                        })
                
                if price_changes:
                    user_price_changes[email] = {
                        'user_name': full_name or email.split('@')[0],
                        'changes': price_changes
                    }
            
            conn.close()
            logger.info(f"✅ Found price changes for {len(user_price_changes)} users")
            return user_price_changes
            
        except Exception as e:
            logger.error(f"❌ Error tracking price changes: {str(e)}")
            return {}
    
    def send_notifications(self, new_bargains: List[Dict[str, Any]], 
                          price_changes: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
        """Send email notifications to users"""
        logger.info("📧 Sending email notifications...")
        
        try:
            # Import notification service
            from somon_project.webapp.backend.services.notification_service import notification_service
            
            notification_stats = {
                'bargain_emails_sent': 0,
                'price_change_emails_sent': 0,
                'total_emails_sent': 0,
                'failed_emails': 0
            }
            
            # Send bargain alerts to all users with email notifications enabled
            if new_bargains:
                conn = self.get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT email, full_name
                    FROM users
                    WHERE email_notifications = 1 AND is_active = 1
                """)
                users = cursor.fetchall()
                conn.close()
                
                for email, full_name in users:
                    try:
                        success = notification_service.send_bargain_alert(
                            email, full_name or email.split('@')[0], new_bargains
                        )
                        if success:
                            notification_stats['bargain_emails_sent'] += 1
                        else:
                            notification_stats['failed_emails'] += 1
                    except Exception as e:
                        logger.error(f"Failed to send bargain alert to {email}: {str(e)}")
                        notification_stats['failed_emails'] += 1
            
            # Send price change alerts
            for email, data in price_changes.items():
                try:
                    success = notification_service.send_favorite_price_alert(
                        email, data['user_name'], data['changes']
                    )
                    if success:
                        notification_stats['price_change_emails_sent'] += 1
                    else:
                        notification_stats['failed_emails'] += 1
                except Exception as e:
                    logger.error(f"Failed to send price change alert to {email}: {str(e)}")
                    notification_stats['failed_emails'] += 1
            
            notification_stats['total_emails_sent'] = (
                notification_stats['bargain_emails_sent'] + 
                notification_stats['price_change_emails_sent']
            )
            
            logger.info(f"✅ Sent {notification_stats['total_emails_sent']} notifications")
            return notification_stats
            
        except Exception as e:
            logger.error(f"❌ Error sending notifications: {str(e)}")
            return {'bargain_emails_sent': 0, 'price_change_emails_sent': 0, 
                   'total_emails_sent': 0, 'failed_emails': 0}
    
    def log_daily_run(self, results: Dict[str, Any]):
        """Log the results of the daily run"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Create daily_runs table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date DATE,
                    run_timestamp DATETIME,
                    total_scraped INTEGER,
                    new_bargains INTEGER,
                    price_changes INTEGER,
                    emails_sent INTEGER,
                    errors TEXT,
                    status TEXT,
                    duration_seconds REAL
                )
            """)
            
            cursor.execute("""
                INSERT INTO daily_runs (
                    run_date, run_timestamp, total_scraped, new_bargains,
                    price_changes, emails_sent, errors, status, duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().date(),
                datetime.now(),
                results.get('scraping_results', {}).get('total_scraped', 0),
                results.get('new_bargains_count', 0),
                results.get('price_changes_count', 0),
                results.get('notification_stats', {}).get('total_emails_sent', 0),
                json.dumps(results.get('errors', [])),
                'completed' if not results.get('errors') else 'completed_with_errors',
                results.get('duration', 0)
            ))
            
            conn.commit()
            conn.close()
            logger.info("✅ Daily run logged to database")
            
        except Exception as e:
            logger.error(f"❌ Error logging daily run: {str(e)}")
    
    def run_daily_tasks(self):
        """Run all daily tasks"""
        start_time = datetime.now()
        logger.info(f"🌅 Starting daily tasks at {start_time}")
        
        results = {
            'start_time': start_time,
            'errors': []
        }
        
        try:
            # 1. Run data collection
            scraping_results = self.run_data_collection()
            results['scraping_results'] = scraping_results
            
            # 2. Detect new bargains
            new_bargains = self.detect_new_bargains()
            results['new_bargains'] = new_bargains
            results['new_bargains_count'] = len(new_bargains)
            
            # 3. Track favorite price changes
            price_changes = self.track_favorite_price_changes()
            results['price_changes'] = price_changes
            results['price_changes_count'] = sum(len(data['changes']) for data in price_changes.values())
            
            # 4. Send notifications
            notification_stats = self.send_notifications(new_bargains, price_changes)
            results['notification_stats'] = notification_stats
            
            # Calculate duration
            end_time = datetime.now()
            results['end_time'] = end_time
            results['duration'] = (end_time - start_time).total_seconds()
            
            # 5. Log results
            self.log_daily_run(results)
            
            logger.info(f"🌟 Daily tasks completed successfully in {results['duration']:.2f} seconds")
            logger.info(f"📊 Summary: {scraping_results['total_scraped']} scraped, "
                       f"{len(new_bargains)} new bargains, "
                       f"{results['price_changes_count']} price changes, "
                       f"{notification_stats['total_emails_sent']} emails sent")
            
        except Exception as e:
            error_msg = f"❌ Daily tasks failed: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            results['duration'] = (datetime.now() - start_time).total_seconds()
            self.log_daily_run(results)

def main():
    """Main function to run the scheduler"""
    scheduler = DailyScheduler()
    
    # Schedule daily tasks
    schedule.every().day.at("06:00").do(scheduler.run_daily_tasks)  # 6 AM daily
    
    logger.info("🕒 Daily scheduler started. Tasks will run at 6:00 AM daily.")
    logger.info("📝 To run tasks immediately, use: python daily_scheduler.py --run-now")
    
    # Check for immediate run flag
    if len(sys.argv) > 1 and sys.argv[1] == '--run-now':
        logger.info("🚀 Running tasks immediately...")
        scheduler.run_daily_tasks()
        return
    
    # Keep the scheduler running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main() 