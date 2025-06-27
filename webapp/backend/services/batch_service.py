#!/usr/bin/env python3
"""
Daily Batch Processing Service for Real Estate Platform

Automated daily scraping, bargain detection, and user notifications.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from ..models.enhanced_database import (
    User, UserAlert, PropertyListing, DataUpdateLog, 
    UserNotification, MarketStats
)
from ..services.notification_service import notification_service
from ..utils.database import get_db, SessionLocal
from ..utils.config import get_settings
import pandas as pd
import joblib
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DailyBatchService:
    """Service for daily batch processing and notifications"""
    
    def __init__(self):
        self.settings = get_settings()
        self.scheduler = AsyncIOScheduler()
        self.is_running = False
        
    def start_scheduler(self):
        """Start the scheduled tasks"""
        if self.is_running:
            return
            
        # Daily batch processing at specified hour
        self.scheduler.add_job(
            self.run_daily_batch,
            CronTrigger(hour=self.settings.DAILY_NOTIFICATION_HOUR, minute=0),
            id='daily_batch',
            name='Daily Batch Processing',
            replace_existing=True
        )
        
        # Hourly light processing for new properties
        self.scheduler.add_job(
            self.run_hourly_check,
            CronTrigger(minute=0),
            id='hourly_check',
            name='Hourly Property Check',
            replace_existing=True
        )
        
        self.scheduler.start()
        self.is_running = True
        logger.info("Daily batch scheduler started")
        
    def stop_scheduler(self):
        """Stop the scheduled tasks"""
        if self.scheduler.running:
            self.scheduler.shutdown()
        self.is_running = False
        logger.info("Daily batch scheduler stopped")
    
    async def run_daily_batch(self):
        """Run the complete daily batch processing"""
        logger.info("Starting daily batch processing")
        
        db = SessionLocal()
        try:
            # Create update log entry
            update_log = DataUpdateLog(
                update_type="daily_batch",
                status="running",
                started_at=datetime.utcnow()
            )
            db.add(update_log)
            db.commit()
            
            try:
                # Step 1: Scrape new properties
                logger.info("Step 1: Scraping new properties")
                new_properties = await self.scrape_properties(db)
                
                # Step 2: Run ML predictions on new properties
                logger.info("Step 2: Running ML predictions")
                predicted_properties = await self.predict_property_prices(db, new_properties)
                
                # Step 3: Identify bargains
                logger.info("Step 3: Identifying bargains")
                bargain_properties = await self.identify_bargains(db, predicted_properties)
                
                # Step 4: Update market statistics
                logger.info("Step 4: Updating market statistics")
                await self.update_market_stats(db)
                
                # Step 5: Process user alerts and send notifications
                logger.info("Step 5: Processing user alerts")
                notifications_sent = await self.process_all_user_alerts(db, bargain_properties)
                
                # Update log with success
                update_log.status = "completed"
                update_log.completed_at = datetime.utcnow()
                update_log.properties_processed = len(new_properties)
                update_log.new_properties = len(new_properties)
                update_log.updated_properties = len(predicted_properties)
                update_log.bargains_found = len(bargain_properties)
                update_log.metadata = {
                    "notifications_sent": notifications_sent,
                    "processing_time_minutes": (datetime.utcnow() - update_log.started_at).total_seconds() / 60
                }
                
                logger.info(f"Daily batch completed successfully: {len(new_properties)} new properties, {len(bargain_properties)} bargains, {notifications_sent} notifications sent")
                
            except Exception as e:
                # Update log with error
                update_log.status = "failed"
                update_log.completed_at = datetime.utcnow()
                update_log.error_message = str(e)
                logger.error(f"Daily batch processing failed: {e}")
                
            db.commit()
            
        finally:
            db.close()
    
    async def run_hourly_check(self):
        """Run hourly check for urgent notifications"""
        logger.info("Starting hourly property check")
        
        db = SessionLocal()
        try:
            # Get properties added in the last hour
            hour_ago = datetime.utcnow() - timedelta(hours=1)
            recent_properties = db.query(PropertyListing).filter(
                PropertyListing.scraped_at >= hour_ago,
                PropertyListing.is_active == True
            ).all()
            
            if not recent_properties:
                logger.info("No new properties in the last hour")
                return
            
            # Check for urgent bargains (excellent category)
            urgent_bargains = [p for p in recent_properties if p.bargain_category == 'excellent']
            
            if urgent_bargains:
                logger.info(f"Found {len(urgent_bargains)} urgent bargains")
                
                # Send instant notifications to users with instant frequency
                await self.send_instant_notifications(db, urgent_bargains)
                
        finally:
            db.close()
    
    async def scrape_properties(self, db: Session) -> List[PropertyListing]:
        """Scrape new properties (placeholder - integrate with existing scraper)"""
        # This would integrate with your existing scraping logic
        # For now, return empty list as scraping is handled separately
        logger.info("Property scraping would be triggered here")
        
        # Get recently added properties (last 24 hours) as "new"
        yesterday = datetime.utcnow() - timedelta(days=1)
        new_properties = db.query(PropertyListing).filter(
            PropertyListing.scraped_at >= yesterday,
            PropertyListing.predicted_price.is_(None)  # Not yet processed
        ).all()
        
        return new_properties
    
    async def predict_property_prices(self, db: Session, properties: List[PropertyListing]) -> List[PropertyListing]:
        """Run ML predictions on properties"""
        if not properties:
            return []
            
        try:
            # Load the trained model
            model_path = self.settings.MODEL_PATH
            model = joblib.load(model_path)
            
            predicted_properties = []
            
            for prop in properties:
                try:
                    # Prepare features for prediction
                    features = self.prepare_features_for_prediction(prop)
                    
                    if features is not None:
                        # Make prediction
                        prediction = model.predict([features])[0]
                        
                        # Update property with prediction
                        prop.predicted_price = float(prediction)
                        prop.price_difference = prop.price - prop.predicted_price
                        prop.price_difference_percentage = (prop.price_difference / prop.predicted_price) * 100
                        
                        predicted_properties.append(prop)
                        
                except Exception as e:
                    logger.error(f"Error predicting price for property {prop.id}: {e}")
                    continue
            
            # Commit predictions
            db.commit()
            logger.info(f"Predicted prices for {len(predicted_properties)} properties")
            
            return predicted_properties
            
        except Exception as e:
            logger.error(f"Error loading model or making predictions: {e}")
            return []
    
    def prepare_features_for_prediction(self, prop: PropertyListing) -> List[float]:
        """Prepare property features for ML prediction"""
        try:
            # This should match your model's feature engineering
            features = []
            
            # Basic features
            features.append(prop.area or 0)
            features.append(prop.rooms or 0)
            features.append(prop.floor or 0)
            features.append(prop.total_floors or 0)
            
            # Encoded categorical features would be added here
            # This is a simplified version - you'd need to use the same
            # encoding as during training
            
            # City encoding (simplified)
            city_mapping = {'Dushanbe': 1, 'Khujand': 2, 'Kulob': 3}
            features.append(city_mapping.get(prop.city, 0))
            
            # Property type encoding
            type_mapping = {'apartment': 1, 'house': 2, 'commercial': 3}
            features.append(type_mapping.get(prop.property_type, 0))
            
            return features if len(features) > 0 else None
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    async def identify_bargains(self, db: Session, properties: List[PropertyListing]) -> List[PropertyListing]:
        """Identify bargain properties based on price differences"""
        bargains = []
        
        for prop in properties:
            if prop.price_difference and prop.price_difference < 0:
                # Property is priced below predicted value
                savings_percentage = abs(prop.price_difference_percentage)
                
                if savings_percentage >= 20:
                    prop.bargain_category = 'excellent'
                    prop.bargain_score = min(savings_percentage / 20, 5.0)  # Score out of 5
                    bargains.append(prop)
                elif savings_percentage >= 10:
                    prop.bargain_category = 'good'
                    prop.bargain_score = min(savings_percentage / 10, 3.0)  # Score out of 3
                    bargains.append(prop)
                else:
                    prop.bargain_category = None
                    prop.bargain_score = None
            else:
                prop.bargain_category = None
                prop.bargain_score = None
        
        db.commit()
        logger.info(f"Identified {len(bargains)} bargain properties")
        
        return bargains
    
    async def update_market_stats(self, db: Session):
        """Update market statistics"""
        try:
            # Get active properties
            active_properties = db.query(PropertyListing).filter(
                PropertyListing.is_active == True
            ).all()
            
            if not active_properties:
                return
            
            # Calculate statistics
            prices = [p.price for p in active_properties if p.price]
            
            if not prices:
                return
            
            stats = MarketStats(
                city=None,  # Overall stats
                total_listings=len(active_properties),
                avg_price=np.mean(prices),
                median_price=np.median(prices),
                min_price=min(prices),
                max_price=max(prices),
                avg_price_per_sqm=np.mean([p.price_per_sqm for p in active_properties if p.price_per_sqm]),
                total_bargains=len([p for p in active_properties if p.bargain_category]),
                excellent_bargains=len([p for p in active_properties if p.bargain_category == 'excellent']),
                good_bargains=len([p for p in active_properties if p.bargain_category == 'good'])
            )
            
            # Remove old stats
            db.query(MarketStats).filter(MarketStats.city.is_(None)).delete()
            
            # Add new stats
            db.add(stats)
            db.commit()
            
            logger.info("Market statistics updated")
            
        except Exception as e:
            logger.error(f"Error updating market stats: {e}")
    
    async def process_all_user_alerts(self, db: Session, bargain_properties: List[PropertyListing]) -> int:
        """Process alerts for all users and send notifications"""
        if not bargain_properties:
            return 0
            
        # Get all active users with active alerts
        users_with_alerts = db.query(User).join(UserAlert).filter(
            User.is_active == True,
            UserAlert.is_active == True
        ).distinct().all()
        
        total_notifications = 0
        
        for user in users_with_alerts:
            try:
                # Check user's notification frequency
                if user.notification_frequency == 'instant':
                    continue  # Already handled in hourly check
                elif user.notification_frequency == 'weekly':
                    # Only send on specific day of week
                    if datetime.utcnow().weekday() != 0:  # Monday
                        continue
                
                # Process user alerts
                sent = await notification_service.process_user_alerts(
                    db, user.id, bargain_properties
                )
                
                if sent:
                    total_notifications += 1
                    
            except Exception as e:
                logger.error(f"Error processing alerts for user {user.id}: {e}")
                continue
        
        logger.info(f"Sent notifications to {total_notifications} users")
        return total_notifications
    
    async def send_instant_notifications(self, db: Session, urgent_bargains: List[PropertyListing]):
        """Send instant notifications for urgent bargains"""
        # Get users with instant notification frequency
        instant_users = db.query(User).filter(
            User.is_active == True,
            User.notification_frequency == 'instant',
            User.push_notifications == True
        ).all()
        
        for user in instant_users:
            try:
                await notification_service.send_bargain_alert(db, user, urgent_bargains)
            except Exception as e:
                logger.error(f"Error sending instant notification to user {user.id}: {e}")
    
    async def run_manual_batch(self) -> Dict[str, Any]:
        """Run batch processing manually (for testing/admin)"""
        logger.info("Running manual batch processing")
        
        db = SessionLocal()
        try:
            start_time = datetime.utcnow()
            
            # Run the same process as daily batch
            new_properties = await self.scrape_properties(db)
            predicted_properties = await self.predict_property_prices(db, new_properties)
            bargain_properties = await self.identify_bargains(db, predicted_properties)
            await self.update_market_stats(db)
            notifications_sent = await self.process_all_user_alerts(db, bargain_properties)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = {
                "success": True,
                "properties_processed": len(new_properties),
                "bargains_found": len(bargain_properties),
                "notifications_sent": notifications_sent,
                "processing_time_seconds": processing_time
            }
            
            logger.info(f"Manual batch completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Manual batch processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            db.close()

# Global batch service instance
daily_batch_service = DailyBatchService()
