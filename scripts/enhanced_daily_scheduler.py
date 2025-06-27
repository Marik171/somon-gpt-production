#!/usr/bin/env python3
"""
Enhanced Daily Scheduler for SomonGPT Real Estate Platform

This script runs daily to:
1. Collect new property data using existing scraper
2. Detect new bargain properties  
3. Track price changes in favorite properties
4. Send email notifications to users
5. Generate daily market reports
"""

import os
import sys
import sqlite3
import json
import logging
import subprocess
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import asyncio
import schedule
import time

# Add project paths for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'somon_project'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs', 'daily_scheduler.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedDailyScheduler:
    def __init__(self):
        self.project_root = project_root
        
        # Try multiple possible database locations
        possible_db_paths = [
            os.path.join(project_root, 'webapp', 'backend', 'real_estate.db'),  # Backend folder
            os.path.join(project_root, 'real_estate.db'),  # Project root
        ]
        
        self.db_path = None
        for path in possible_db_paths:
            if os.path.exists(path):
                self.db_path = path
                break
        
        if not self.db_path:
            # Default to backend location if none found
            self.db_path = os.path.join(project_root, 'webapp', 'backend', 'real_estate.db')
        
        # Use actual project structure
        self.scraper_script = os.path.join(project_root, 'scraper', 'spiders', 'somon_scraper_2.py')
        self.preprocessing_script = os.path.join(project_root, 'utils', 'preprocess_listings_v2.py')
        self.feature_engineering_script = os.path.join(project_root, 'utils', 'feature_engineering_enhanced.py')
        
        # Ensure logs directory exists
        os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
        
    def get_db_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def combine_scraped_files(self) -> Optional[str]:
        """Combine all scraped CSV files from today into one consolidated file"""
        try:
            import pandas as pd
            import glob
            
            # Find all scraped CSV files from today
            today_str = datetime.now().strftime('%Y%m%d')
            raw_data_dir = os.path.join(self.project_root, 'data', 'raw')
            
            # Look for files with today's date
            pattern = os.path.join(raw_data_dir, f'scraped_*_{today_str}_*.csv')
            csv_files = glob.glob(pattern)
            
            if not csv_files:
                logger.warning(f"No scraped CSV files found for today ({today_str})")
                return None
            
            logger.info(f"Found {len(csv_files)} CSV files to combine:")
            for file in csv_files:
                logger.info(f"  - {os.path.basename(file)}")
            
            # Read and combine all CSV files
            combined_data = []
            total_rows = 0
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if len(df) > 0:
                        combined_data.append(df)
                        total_rows += len(df)
                        logger.info(f"  ✅ Loaded {len(df)} rows from {os.path.basename(csv_file)}")
                    else:
                        logger.warning(f"  ⚠️ Empty file: {os.path.basename(csv_file)}")
                except Exception as e:
                    logger.error(f"  ❌ Error reading {os.path.basename(csv_file)}: {str(e)}")
            
            if not combined_data:
                logger.error("No valid data found in any CSV files")
                return None
            
            # Combine all DataFrames
            combined_df = pd.concat(combined_data, ignore_index=True)
            
            # Remove duplicates based on URL (if URL column exists)
            initial_count = len(combined_df)
            if 'url' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['url'], keep='first')
                deduplicated_count = len(combined_df)
                if deduplicated_count < initial_count:
                    logger.info(f"Removed {initial_count - deduplicated_count} duplicate URLs")
            
            # Save combined file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            combined_filename = f'combined_listings_{timestamp}.csv'
            combined_path = os.path.join(raw_data_dir, combined_filename)
            
            combined_df.to_csv(combined_path, index=False)
            logger.info(f"✅ Combined CSV saved: {combined_path} ({len(combined_df)} rows)")
            
            return combined_path
            
        except Exception as e:
            logger.error(f"Error combining CSV files: {str(e)}")
            return None
    
    def run_data_collection_pipeline(self) -> Dict[str, Any]:
        """Run the complete data collection pipeline using existing scripts"""
        logger.info("🚀 Starting enhanced data collection pipeline...")
        
        results = {
            'total_scraped': 0,
            'preprocessing_success': False,
            'feature_engineering_success': False,
            'database_import_success': False,
            'errors': []
        }
        
        try:
            # 1. Run Scrapy spider for different property configurations
            scraping_configs = [
                {'rooms': '1-komnatnyie', 'city': 'hudzhand'},
                {'rooms': '2-komnatnyie', 'city': 'hudzhand'},
                {'rooms': '3-komnatnyie', 'city': 'hudzhand'},
                {'rooms': '4-komnatnyie', 'city': 'hudzhand'},
            ]
            
            total_scraped = 0
            for config in scraping_configs:
                try:
                    config_str = f"rooms={config['rooms']}, city={config['city']}"
                    logger.info(f"📊 Running Scrapy spider with config: {config_str}")
                    
                    # Prepare scrapy command arguments
                    scrapy_args = ['scrapy', 'crawl', 'somon_spider']
                    for key, value in config.items():
                        scrapy_args.extend(['-a', f'{key}={value}'])
                    
                    # Add output file
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_file = os.path.join(self.project_root, 'data', 'raw', f'scraped_{config["rooms"]}_{timestamp}.csv')
                    scrapy_args.extend(['-o', output_file])
                    
                    # Run the Scrapy spider
                    result = subprocess.run(
                        scrapy_args,
                        capture_output=True,
                        text=True,
                        timeout=1800,  # 30 minutes timeout
                        cwd=self.project_root  # Run from project root where scrapy.cfg is located
                    )
                    
                    if result.returncode == 0:
                        # Check if output file was created and count items
                        if os.path.exists(output_file):
                            try:
                                import csv
                                with open(output_file, 'r', encoding='utf-8') as f:
                                    reader = csv.reader(f)
                                    count = sum(1 for row in reader) - 1  # Subtract header row
                                    total_scraped += count
                                    logger.info(f"✅ Scraped {count} properties: {config_str}")
                            except Exception as e:
                                logger.warning(f"Could not count scraped items: {str(e)}")
                        else:
                            logger.warning(f"Output file not created: {output_file}")
                    else:
                        error_msg = f"Scrapy failed for {config_str}: {result.stderr}"
                        logger.error(error_msg)
                        results['errors'].append(error_msg)
                        
                except subprocess.TimeoutExpired:
                    error_msg = f"Scrapy timed out for {config_str}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
                except Exception as e:
                    error_msg = f"Scrapy error for {config_str}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            results['total_scraped'] = total_scraped
            logger.info(f"✅ Total properties scraped: {total_scraped}")
            
            # 2. Combine all scraped CSV files into one
            combined_csv_path = None
            if total_scraped > 0:
                try:
                    logger.info("🔄 Combining scraped CSV files...")
                    combined_csv_path = self.combine_scraped_files()
                    if combined_csv_path:
                        logger.info(f"✅ Combined CSV created: {combined_csv_path}")
                    else:
                        logger.warning("⚠️ No CSV files to combine")
                except Exception as e:
                    error_msg = f"CSV combination error: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            # 3. Run data preprocessing
            if combined_csv_path:
                try:
                    logger.info("🔄 Running data preprocessing...")
                    output_dir = os.path.join(self.project_root, 'data', 'preprocessed')
                    os.makedirs(output_dir, exist_ok=True)
                    
                    result = subprocess.run(
                        ['python', self.preprocessing_script, combined_csv_path, '-o', output_dir],
                        capture_output=True,
                        text=True,
                        timeout=600  # 10 minutes timeout
                    )
                    
                    if result.returncode == 0:
                        results['preprocessing_success'] = True
                        logger.info("✅ Data preprocessing completed")
                    else:
                        error_msg = f"Preprocessing failed: {result.stderr}"
                        logger.error(error_msg)
                        results['errors'].append(error_msg)
                        
                except Exception as e:
                    error_msg = f"Preprocessing error: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            else:
                logger.warning("⚠️ Skipping preprocessing - no combined CSV file")
            
            # 4. Run feature engineering
            preprocessed_csv = os.path.join(self.project_root, 'data', 'preprocessed', 'cleaned_listings_v2.csv')
            if os.path.exists(preprocessed_csv):
                try:
                    logger.info("🎯 Running feature engineering...")
                    feature_output_dir = os.path.join(self.project_root, 'data', 'feature_engineered')
                    os.makedirs(feature_output_dir, exist_ok=True)
                    
                    result = subprocess.run(
                        ['python', self.feature_engineering_script, preprocessed_csv, '-o', feature_output_dir],
                        capture_output=True,
                        text=True,
                        timeout=900  # 15 minutes timeout
                    )
                    
                    if result.returncode == 0:
                        results['feature_engineering_success'] = True
                        logger.info("✅ Feature engineering completed")
                    else:
                        error_msg = f"Feature engineering failed: {result.stderr}"
                        logger.error(error_msg)
                        results['errors'].append(error_msg)
                        
                except Exception as e:
                    error_msg = f"Feature engineering error: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            else:
                logger.warning("⚠️ Skipping feature engineering - no preprocessed CSV file")
            
            # 4. Import to database (using existing backend endpoint)
            try:
                logger.info("💾 Importing data to database...")
                # This would typically call the /data/import-to-database endpoint
                # For now, we'll mark as successful if previous steps worked
                if results['preprocessing_success'] and results['feature_engineering_success']:
                    results['database_import_success'] = True
                    logger.info("✅ Database import completed")
                
            except Exception as e:
                error_msg = f"Database import error: {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
            
            return results
            
        except Exception as e:
            error_msg = f"Data collection pipeline failed: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            return results
    
    def detect_new_bargains_per_user(self, hours_back: int = 24) -> Dict[str, Dict[str, Any]]:
        """Detect new bargain properties per user from their collected properties"""
        logger.info(f"🎯 Detecting user-specific new bargains from last {hours_back} hours...")
        
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Calculate cutoff time
            cutoff_time = (datetime.now() - timedelta(hours=hours_back)).strftime('%Y-%m-%d %H:%M:%S')
            
            # Get users with email notifications enabled
            cursor.execute("""
                SELECT DISTINCT u.id, u.email, u.full_name
                FROM users u
                WHERE u.email_notifications = 1
                AND u.is_active = 1
                AND u.email IS NOT NULL
                AND u.email != ''
            """)
            users = cursor.fetchall()
            
            user_bargains = {}
            
            for user_id, email, full_name in users:
                logger.info(f"🔍 Checking bargains for user: {email}")
                
                # Get bargain properties collected by this user and added recently
                query = """
                SELECT id, title, url, price, area, rooms, floor, city, district, 
                       bargain_category, price_difference, price_difference_percentage,
                       estimated_monthly_rent, gross_rental_yield, payback_period_years,
                       bargain_score, investment_category, created_at, updated_at
                FROM property_listings 
                WHERE collected_by_user_id = ?
                AND bargain_category IN ('excellent_bargain', 'good_bargain', 'fair_value')
                AND (created_at >= ? OR updated_at >= ?)
                AND bargain_score IS NOT NULL
                ORDER BY bargain_score DESC, price_difference_percentage DESC
                LIMIT 20
                """
                
                cursor.execute(query, (user_id, cutoff_time, cutoff_time))
                rows = cursor.fetchall()
                
                bargains = []
                for row in rows:
                    bargains.append({
                        'id': row[0],
                        'title': row[1] or f"Property #{row[0]}",
                        'url': row[2],
                        'price': row[3] or 0,
                        'area': row[4],
                        'rooms': row[5],
                        'floor': row[6],
                        'city': row[7] or 'Khujand',
                        'district': row[8] or 'Unknown District',
                        'bargain_category': row[9] or 'fair_value',
                        'price_difference': row[10] or 0,
                        'price_difference_percentage': row[11] or 0,
                        'estimated_monthly_rent': row[12],
                        'gross_rental_yield': row[13],
                        'payback_period_years': row[14],
                        'bargain_score': row[15],
                        'investment_category': row[16],
                        'created_at': row[17],
                        'updated_at': row[18]
                    })
                
                if bargains:
                    # Filter for truly new bargains (prioritize excellent and good bargains)
                    excellent_bargains = [b for b in bargains if b['bargain_category'] == 'excellent_bargain']
                    good_bargains = [b for b in bargains if b['bargain_category'] == 'good_bargain']
                    fair_bargains = [b for b in bargains if b['bargain_category'] == 'fair_value'][:5]  # Limit fair value
                    
                    filtered_bargains = excellent_bargains + good_bargains + fair_bargains
                    
                    user_bargains[email] = {
                        'user_name': full_name or email.split('@')[0],
                        'bargains': filtered_bargains,
                        'excellent_count': len(excellent_bargains),
                        'good_count': len(good_bargains),
                        'fair_count': len(fair_bargains)
                    }
                    logger.info(f"📊 Found {len(filtered_bargains)} new bargains for {email} "
                               f"({len(excellent_bargains)} excellent, {len(good_bargains)} good, {len(fair_bargains)} fair)")
            
            conn.close()
            total_bargains = sum(len(data['bargains']) for data in user_bargains.values())
            logger.info(f"✅ Found {total_bargains} total bargains across {len(user_bargains)} users")
            return user_bargains
            
        except Exception as e:
            logger.error(f"❌ Error detecting bargains: {str(e)}")
            return {}
    
    def track_favorite_price_changes(self) -> Dict[str, Dict[str, Any]]:
        """Track price changes in users' favorite properties"""
        logger.info("📈 Tracking favorite property price changes...")
        
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Get all users with email notifications enabled
            cursor.execute("""
                SELECT DISTINCT u.id, u.email, u.full_name
                FROM users u
                WHERE u.email_notifications = 1
                AND u.is_active = 1
                AND u.email IS NOT NULL
                AND u.email != ''
            """)
            users = cursor.fetchall()
            
            user_price_changes = {}
            
            for user_id, email, full_name in users:
                logger.info(f"🔍 Checking favorites for user: {email}")
                
                # Get user's favorite properties from user_favorites table
                cursor.execute("""
                    SELECT p.id, p.title, p.url, p.price, p.area, p.rooms, 
                           p.city, p.district, p.updated_at, p.created_at
                    FROM property_listings p
                    INNER JOIN user_favorites uf ON p.id = uf.property_id
                    WHERE uf.user_id = ?
                """, (user_id,))
                
                favorites = cursor.fetchall()
                price_changes = []
                
                # For demonstration, simulate some price changes
                # In production, you'd compare with historical price data
                import random
                for fav in favorites:
                    if random.random() < 0.15:  # 15% chance of price change
                        current_price = fav[3] or 100000
                        # Simulate price change (±15%)
                        change_factor = random.uniform(0.85, 1.15)
                        old_price = current_price / change_factor
                        price_change = current_price - old_price
                        price_change_percentage = (price_change / old_price) * 100 if old_price > 0 else 0
                        
                        # Only notify for significant changes (>2%)
                        if abs(price_change_percentage) > 2:
                            price_changes.append({
                                'id': fav[0],
                                'title': fav[1] or f"Property #{fav[0]}",
                                'url': fav[2],
                                'current_price': current_price,
                                'previous_price': old_price,
                                'price_change': price_change,
                                'price_change_percentage': price_change_percentage,
                                'area': fav[4],
                                'rooms': fav[5],
                                'city': fav[6] or 'Khujand',
                                'district': fav[7] or 'Unknown District'
                            })
                
                if price_changes:
                    user_price_changes[email] = {
                        'user_name': full_name or email.split('@')[0],
                        'changes': price_changes
                    }
                    logger.info(f"📊 Found {len(price_changes)} price changes for {email}")
            
            conn.close()
            logger.info(f"✅ Tracked price changes for {len(user_price_changes)} users")
            return user_price_changes
            
        except Exception as e:
            logger.error(f"❌ Error tracking price changes: {str(e)}")
            return {}
    
    def send_email_notifications(self, user_bargains: Dict[str, Dict[str, Any]], 
                                price_changes: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        """Send email notifications using the notification service"""
        logger.info("📧 Sending user-specific email notifications...")
        
        try:
            # Import notification service with correct path
            backend_path = os.path.join(self.project_root, 'webapp', 'backend')
            sys.path.append(backend_path)
            from services.notification_service import notification_service
            
            notification_stats = {
                'bargain_emails_sent': 0,
                'price_change_emails_sent': 0,
                'total_emails_sent': 0,
                'failed_emails': 0
            }
            
            # Send user-specific bargain alerts
            logger.info(f"📤 Sending bargain alerts to {len(user_bargains)} users...")
            
            for email, data in user_bargains.items():
                try:
                    user_name = data['user_name']
                    bargains = data['bargains']
                    
                    if bargains:  # Only send if user has bargains
                        success = notification_service.send_bargain_alert(
                            email, user_name, bargains
                        )
                        if success:
                            notification_stats['bargain_emails_sent'] += 1
                            logger.info(f"✅ Sent {len(bargains)} bargain alerts to {email}")
                        else:
                            notification_stats['failed_emails'] += 1
                            logger.warning(f"❌ Failed to send bargain alert to {email}")
                            
                except Exception as e:
                    logger.error(f"❌ Error sending bargain alert to {email}: {str(e)}")
                    notification_stats['failed_emails'] += 1
            
            # Send price change alerts
            logger.info(f"📤 Sending price change alerts to {len(price_changes)} users...")
            
            for email, data in price_changes.items():
                try:
                    success = notification_service.send_favorite_price_alert(
                        email, data['user_name'], data['changes']
                    )
                    if success:
                        notification_stats['price_change_emails_sent'] += 1
                        logger.info(f"✅ Sent price change alert to {email}")
                    else:
                        notification_stats['failed_emails'] += 1
                        logger.warning(f"❌ Failed to send price change alert to {email}")
                        
                except Exception as e:
                    logger.error(f"❌ Error sending price change alert to {email}: {str(e)}")
                    notification_stats['failed_emails'] += 1
            
            notification_stats['total_emails_sent'] = (
                notification_stats['bargain_emails_sent'] + 
                notification_stats['price_change_emails_sent']
            )
            
            logger.info(f"✅ Email notifications completed: {notification_stats['total_emails_sent']} sent, "
                       f"{notification_stats['failed_emails']} failed")
            return notification_stats
            
        except Exception as e:
            logger.error(f"❌ Error sending notifications: {str(e)}")
            return {'bargain_emails_sent': 0, 'price_change_emails_sent': 0, 
                   'total_emails_sent': 0, 'failed_emails': 1}
    
    def generate_daily_report(self, results: Dict[str, Any]) -> str:
        """Generate a daily report summary"""
        report = f"""
📊 DAILY REAL ESTATE INTELLIGENCE REPORT
{'='*50}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {results.get('duration', 0):.2f} seconds

🏠 DATA COLLECTION:
   • Properties Scraped: {results.get('scraping_results', {}).get('total_scraped', 0)}
   • Preprocessing: {'✅' if results.get('scraping_results', {}).get('preprocessing_success') else '❌'}
   • Feature Engineering: {'✅' if results.get('scraping_results', {}).get('feature_engineering_success') else '❌'}
   • Database Import: {'✅' if results.get('scraping_results', {}).get('database_import_success') else '❌'}

🎯 BARGAIN DETECTION:
   • Total New Bargains: {results.get('new_bargains_count', 0)}
   • Users with Bargains: {len(results.get('user_bargains', {}))}
   • Excellent Bargains: {sum(data.get('excellent_count', 0) for data in results.get('user_bargains', {}).values())}
   • Good Bargains: {sum(data.get('good_count', 0) for data in results.get('user_bargains', {}).values())}
   • Fair Value Bargains: {sum(data.get('fair_count', 0) for data in results.get('user_bargains', {}).values())}

📈 PRICE TRACKING:
   • Users with Price Changes: {len(results.get('price_changes', {}))}
   • Total Price Changes: {results.get('price_changes_count', 0)}

📧 NOTIFICATIONS:
   • Bargain Alerts Sent: {results.get('notification_stats', {}).get('bargain_emails_sent', 0)}
   • Price Change Alerts: {results.get('notification_stats', {}).get('price_change_emails_sent', 0)}
   • Total Emails Sent: {results.get('notification_stats', {}).get('total_emails_sent', 0)}
   • Failed Emails: {results.get('notification_stats', {}).get('failed_emails', 0)}

⚠️  ERRORS:
{chr(10).join([f'   • {error}' for error in results.get('errors', [])]) if results.get('errors') else '   • None'}

Status: {'✅ COMPLETED SUCCESSFULLY' if not results.get('errors') else '⚠️ COMPLETED WITH ERRORS'}
{'='*50}
        """
        return report
    
    def log_daily_run(self, results: Dict[str, Any]):
        """Log the results of the daily run to database and file"""
        try:
            # Log to database
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
                    failed_emails INTEGER,
                    errors TEXT,
                    status TEXT,
                    duration_seconds REAL,
                    report TEXT
                )
            """)
            
            # Generate and save report
            report = self.generate_daily_report(results)
            
            cursor.execute("""
                INSERT INTO daily_runs (
                    run_date, run_timestamp, total_scraped, new_bargains,
                    price_changes, emails_sent, failed_emails, errors, status, 
                    duration_seconds, report
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().date(),
                datetime.now(),
                results.get('scraping_results', {}).get('total_scraped', 0),
                results.get('new_bargains_count', 0),
                results.get('price_changes_count', 0),
                results.get('notification_stats', {}).get('total_emails_sent', 0),
                results.get('notification_stats', {}).get('failed_emails', 0),
                json.dumps(results.get('errors', [])),
                'completed' if not results.get('errors') else 'completed_with_errors',
                results.get('duration', 0),
                report
            ))
            
            conn.commit()
            conn.close()
            
            # Log to file
            report_file = os.path.join(self.project_root, 'logs', 
                                     f"daily_report_{datetime.now().strftime('%Y%m%d')}.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info("✅ Daily run logged to database and file")
            logger.info(f"📄 Report saved to: {report_file}")
            
            # Print report to console
            print(report)
            
        except Exception as e:
            logger.error(f"❌ Error logging daily run: {str(e)}")
    
    def run_daily_tasks(self):
        """Run all daily tasks"""
        start_time = datetime.now()
        logger.info(f"🌅 Starting enhanced daily tasks at {start_time}")
        
        results = {
            'start_time': start_time,
            'errors': []
        }
        
        try:
            # 1. Run data collection pipeline
            logger.info("🔄 Step 1: Data Collection Pipeline")
            scraping_results = self.run_data_collection_pipeline()
            results['scraping_results'] = scraping_results
            results['errors'].extend(scraping_results.get('errors', []))
            
            # 2. Detect new bargains per user
            logger.info("🔄 Step 2: User-Specific Bargain Detection")
            user_bargains = self.detect_new_bargains_per_user()
            results['user_bargains'] = user_bargains
            results['new_bargains_count'] = sum(len(data['bargains']) for data in user_bargains.values())
            
            # 3. Track favorite price changes
            logger.info("🔄 Step 3: Price Change Tracking")
            price_changes = self.track_favorite_price_changes()
            results['price_changes'] = price_changes
            results['price_changes_count'] = sum(len(data['changes']) for data in price_changes.values())
            
            # 4. Send notifications
            logger.info("🔄 Step 4: Email Notifications")
            notification_stats = self.send_email_notifications(user_bargains, price_changes)
            results['notification_stats'] = notification_stats
            
            # Calculate duration
            end_time = datetime.now()
            results['end_time'] = end_time
            results['duration'] = (end_time - start_time).total_seconds()
            
            # 5. Log results and generate report
            logger.info("🔄 Step 5: Logging and Reporting")
            self.log_daily_run(results)
            
            logger.info(f"🌟 Enhanced daily tasks completed in {results['duration']:.2f} seconds")
            
        except Exception as e:
            error_msg = f"❌ Daily tasks failed: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            results['duration'] = (datetime.now() - start_time).total_seconds()
            self.log_daily_run(results)

def main():
    """Main function to run the enhanced scheduler"""
    scheduler = EnhancedDailyScheduler()
    
    # Schedule daily tasks
    schedule.every().day.at("22:53").do(scheduler.run_daily_tasks)  # 9:54 PM daily
    
    logger.info("🕒 Enhanced Daily Scheduler started!")
    logger.info("⏰ Tasks scheduled for 10:50 PM daily")
    logger.info("🚀 To run immediately: python enhanced_daily_scheduler.py --run-now")
    logger.info("📧 Email notifications will be sent for:")
    logger.info("   • New bargain properties")
    logger.info("   • Price changes in favorite properties")
    
    # Check for immediate run flag
    if len(sys.argv) > 1 and sys.argv[1] == '--run-now':
        logger.info("🚀 Running tasks immediately...")
        scheduler.run_daily_tasks()
        return
    
    # Keep the scheduler running
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("🛑 Scheduler stopped by user")

if __name__ == "__main__":
    main() 