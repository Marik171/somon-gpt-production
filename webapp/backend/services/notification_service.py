#!/usr/bin/env python3
"""
Notification Service for Real Estate Platform

Email and push notification service for user alerts and bargain notifications.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import json
from jinja2 import Template
from cryptography.fernet import Fernet
import base64

# Configure logging
logger = logging.getLogger(__name__)

class EmailNotificationService:
    def __init__(self):
        # Try to load email configuration from config file first, then environment variables
        self.smtp_server = None
        self.smtp_port = None
        self.sender_email = None
        self.sender_password = None
        self.sender_name = None
        self.is_configured = False
        
        # Try to load from config file
        # Go up from webapp/backend/services/ to the project root, then to config/
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        config_path = os.path.join(project_root, 'config', 'email_config.json')
        logger.info(f"🔍 Looking for config at: {config_path}")
        if os.path.exists(config_path):
            logger.info("📄 Config file found, loading...")
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                logger.info(f"📧 Email user: {config.get('email_user')}")
                
                self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
                self.smtp_port = int(config.get('smtp_port', 587))
                self.sender_email = config.get('email_user')
                self.sender_name = config.get('from_name', 'SomonGPT Real Estate Intelligence')
                
                # Decrypt password if it's encrypted
                encrypted_password = config.get('email_password')
                if encrypted_password:
                    logger.info("🔐 Attempting to decrypt password...")
                    self.sender_password = self._decrypt_password(encrypted_password)
                    logger.info(f"🔐 Password decrypted: {self.sender_password is not None}")
                
                if self.sender_email and self.sender_password:
                    self.is_configured = True
                    logger.info(f"✅ Email configuration loaded from config file: {self.sender_email}")
                else:
                    logger.warning(f"⚠️  Email config incomplete - email: {bool(self.sender_email)}, password: {bool(self.sender_password)}")
                    
            except Exception as e:
                logger.error(f"❌ Error loading email config file: {str(e)}")
        else:
            logger.warning(f"⚠️  Config file not found at: {config_path}")
        
        # Fall back to environment variables if config file not available or incomplete
        if not self.is_configured:
            self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
            self.sender_email = os.getenv('SENDER_EMAIL')
            self.sender_password = os.getenv('SENDER_PASSWORD')
            self.sender_name = os.getenv('SENDER_NAME', 'SomonGPT Real Estate Intelligence')
            
            if self.sender_email and self.sender_password:
                self.is_configured = True
                logger.info(f"✅ Email configuration loaded from environment variables: {self.sender_email}")
            else:
                logger.warning("⚠️  Email not configured. Set environment variables or config file.")
    
    def _decrypt_password(self, encrypted_password: str) -> str:
        """Decrypt the encrypted email password using the key file"""
        try:
            # Look for the encryption key file
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            key_path = os.path.join(project_root, 'config', 'email_key.key')
            
            if not os.path.exists(key_path):
                logger.error(f"❌ Encryption key file not found at: {key_path}")
                logger.info("💡 Run 'python scripts/configure_email_once.py' to set up email configuration")
                return None
            
            # Load the encryption key
            with open(key_path, 'rb') as f:
                key = f.read()
            
            # Decrypt the password
            fernet = Fernet(key)
            decrypted = fernet.decrypt(encrypted_password.encode())
            decrypted_password = decrypted.decode()
            
            logger.info("✅ Email password successfully decrypted")
            return decrypted_password
            
        except Exception as e:
            logger.error(f"❌ Error decrypting password: {str(e)}")
            logger.info("💡 Run 'python scripts/configure_email_once.py' to reconfigure email")
            return None
        
    def send_email(self, recipient_email: str, subject: str, html_content: str, text_content: str = None) -> bool:
        """Send an email with HTML content"""
        if not self.is_configured:
            logger.error("❌ Email service not configured. Cannot send email.")
            return False
            
        try:
            # Create message container
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.sender_name} <{self.sender_email}>"
            msg['To'] = recipient_email
            
            # Create the plain-text and HTML version of your message
            if text_content:
                part1 = MIMEText(text_content, 'plain')
                msg.attach(part1)
            
            part2 = MIMEText(html_content, 'html')
            msg.attach(part2)
            
            # Create secure connection and send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
                
            logger.info(f"✅ Email sent successfully to {recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send email to {recipient_email}: {str(e)}")
            return False
    
    def send_bargain_alert(self, recipient_email: str, user_name: str, new_bargains: List[Dict[str, Any]]) -> bool:
        """Send notification about new bargain properties"""
        if not new_bargains:
            return True
            
        subject = f"🎯 New Bargain Properties Found - {len(new_bargains)} Opportunities!"
        
        # HTML template for bargain alert
        html_template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>New Bargain Properties</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: linear-gradient(135deg, #2563eb, #10b981); color: white; padding: 20px; border-radius: 8px; text-align: center; }
                .property-card { border: 1px solid #e0e0e0; border-radius: 8px; margin: 15px 0; padding: 15px; background: #f9f9f9; }
                .price { font-size: 18px; font-weight: bold; color: #10b981; }
                .bargain-badge { background: #ef4444; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
                .excellent-badge { background: #10b981; }
                .good-badge { background: #3b82f6; }
                .footer { margin-top: 30px; padding: 20px; background: #f5f5f5; border-radius: 8px; text-align: center; }
                .cta-button { background: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block; margin: 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 New Bargain Properties Found!</h1>
                    <p>Hello {{ user_name }}, we've discovered {{ bargain_count }} new investment opportunities for you!</p>
                </div>
                
                <div style="margin: 20px 0;">
                    <h2>🏠 Latest Bargain Properties</h2>
                    {% for property in properties %}
                    <div class="property-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin: 0;">{{ property.title or 'Property #' + property.id|string }}</h3>
                            <span class="bargain-badge {% if property.bargain_category == 'excellent_bargain' %}excellent-badge{% elif property.bargain_category == 'good_bargain' %}good-badge{% endif %}">
                                {{ property.bargain_category.replace('_', ' ').title() }}
                            </span>
                        </div>
                        
                        <div style="margin: 10px 0;">
                            <div class="price">{{ "{:,.0f}".format(property.price) }} TJS</div>
                            <div style="color: #666;">
                                📍 {{ property.district or 'Unknown District' }}, {{ property.city or 'Khujand' }}<br>
                                🏠 {{ property.rooms or 'N/A' }} rooms, {{ property.area or 'N/A' }} m²<br>
                                💰 Potential Savings: {{ "{:,.0f}".format(property.price_difference if property.price_difference is not none else 0) }} TJS ({{ "{:.1f}".format(property.price_difference_percentage if property.price_difference_percentage is not none else 0) }}%)
                            </div>
                        </div>
                        
                        {% if property.estimated_monthly_rent %}
                        <div style="background: #e0f2fe; padding: 10px; border-radius: 4px; margin-top: 10px;">
                            <strong>💸 Investment Analysis:</strong><br>
                                                            Monthly Rent: {{ "{:,.0f}".format(property.estimated_monthly_rent if property.estimated_monthly_rent is not none else 0) }} TJS<br>
                                Annual Yield: {{ "{:.1f}".format(property.gross_rental_yield if property.gross_rental_yield is not none else 0) }}%<br>
                                Payback: {{ "{:.1f}".format(property.payback_period_years if property.payback_period_years is not none else 0) }} years
                        </div>
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
                
                <div class="footer">
                    <p><strong>Don't miss out on these opportunities!</strong></p>
                                            <a href="https://somongpt.vercel.app/bargains" class="cta-button">View All Bargains</a>
                        <a href="https://somongpt.vercel.app/search" class="cta-button">Search Properties</a>
                    
                    <p style="margin-top: 20px; font-size: 12px; color: #666;">
                        This is an automated notification from SomonGPT Real Estate Intelligence.<br>
                        To unsubscribe, visit your profile settings.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """)
        
        html_content = html_template.render(
            user_name=user_name,
            bargain_count=len(new_bargains),
            properties=new_bargains
        )
        
        # Plain text version
        text_content = f"""
        New Bargain Properties Found!
        
        Hello {user_name},
        
        We've discovered {len(new_bargains)} new investment opportunities for you:
        
        """
        
        for i, prop in enumerate(new_bargains, 1):
            text_content += f"""
        {i}. {prop.get('title', f'Property #{prop.get("id", "N/A")}')}
           Price: {prop.get('price', 0):,.0f} TJS
           Location: {prop.get('district', 'Unknown')}, {prop.get('city', 'Khujand')}
           Rooms: {prop.get('rooms', 'N/A')}, Area: {prop.get('area', 'N/A')} m²
           Savings: {prop.get('price_difference', 0):,.0f} TJS ({prop.get('price_difference_percentage', 0):.1f}%)
           Category: {prop.get('bargain_category', '').replace('_', ' ').title()}
        """
        
        text_content += f"""
        
        View all bargains: https://somongpt.vercel.app/bargains
        Search properties: https://somongpt.vercel.app/search
        
        Best regards,
        SomonGPT Real Estate Intelligence Team
        """
        
        return self.send_email(recipient_email, subject, html_content, text_content)
    
    def send_favorite_price_alert(self, recipient_email: str, user_name: str, price_changes: List[Dict[str, Any]]) -> bool:
        """Send notification about favorite property price changes"""
        if not price_changes:
            return True
            
        price_drops = [p for p in price_changes if p.get('price_change', 0) < 0]
        price_increases = [p for p in price_changes if p.get('price_change', 0) > 0]
        
        subject = f"📈 Price Changes in Your Favorite Properties - {len(price_drops)} Drops, {len(price_increases)} Increases"
        
        html_template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Favorite Properties Price Changes</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: linear-gradient(135deg, #f59e0b, #ef4444); color: white; padding: 20px; border-radius: 8px; text-align: center; }
                .property-card { border: 1px solid #e0e0e0; border-radius: 8px; margin: 15px 0; padding: 15px; }
                .price-drop { background: #f0fdf4; border-left: 4px solid #10b981; }
                .price-increase { background: #fef2f2; border-left: 4px solid #ef4444; }
                .price-change { font-size: 16px; font-weight: bold; }
                .price-drop .price-change { color: #10b981; }
                .price-increase .price-change { color: #ef4444; }
                .footer { margin-top: 30px; padding: 20px; background: #f5f5f5; border-radius: 8px; text-align: center; }
                .cta-button { background: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block; margin: 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📈 Your Favorite Properties Update</h1>
                    <p>Hello {{ user_name }}, here are the latest price changes in your favorite properties!</p>
                </div>
                
                {% if price_drops %}
                <div style="margin: 20px 0;">
                    <h2 style="color: #10b981;">📉 Price Drops ({{ price_drops|length }})</h2>
                    {% for property in price_drops %}
                    <div class="property-card price-drop">
                        <h3 style="margin: 0;">{{ property.title or 'Property #' + property.id|string }}</h3>
                        <div style="margin: 10px 0;">
                            <div style="font-size: 18px; font-weight: bold;">{{ "{:,.0f}".format(property.current_price) }} TJS</div>
                            <div class="price-change">
                                ⬇️ {{ "{:,.0f}".format(property.price_change * -1) }} TJS ({{ "{:.1f}".format(property.price_change_percentage * -1) }}% decrease)
                            </div>
                            <div style="color: #666; margin-top: 5px;">
                                📍 {{ property.district or 'Unknown District' }}, {{ property.city or 'Khujand' }}<br>
                                🏠 {{ property.rooms or 'N/A' }} rooms, {{ property.area or 'N/A' }} m²<br>
                                Previous Price: {{ "{:,.0f}".format(property.previous_price) }} TJS
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
                
                {% if price_increases %}
                <div style="margin: 20px 0;">
                    <h2 style="color: #ef4444;">📈 Price Increases ({{ price_increases|length }})</h2>
                    {% for property in price_increases %}
                    <div class="property-card price-increase">
                        <h3 style="margin: 0;">{{ property.title or 'Property #' + property.id|string }}</h3>
                        <div style="margin: 10px 0;">
                            <div style="font-size: 18px; font-weight: bold;">{{ "{:,.0f}".format(property.current_price) }} TJS</div>
                            <div class="price-change">
                                ⬆️ {{ "{:,.0f}".format(property.price_change) }} TJS ({{ "{:.1f}".format(property.price_change_percentage) }}% increase)
                            </div>
                            <div style="color: #666; margin-top: 5px;">
                                📍 {{ property.district or 'Unknown District' }}, {{ property.city or 'Khujand' }}<br>
                                🏠 {{ property.rooms or 'N/A' }} rooms, {{ property.area or 'N/A' }} m²<br>
                                Previous Price: {{ "{:,.0f}".format(property.previous_price) }} TJS
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
                
                <div class="footer">
                    <p><strong>Stay updated with your investment opportunities!</strong></p>
                                                                 <a href="https://somongpt.vercel.app/history" class="cta-button">View Your Favorites</a>
                        <a href="https://somongpt.vercel.app/search" class="cta-button">Search More Properties</a>
                    
                    <p style="margin-top: 20px; font-size: 12px; color: #666;">
                        This is an automated notification from SomonGPT Real Estate Intelligence.<br>
                        To unsubscribe, visit your profile settings.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """)
        
        html_content = html_template.render(
            user_name=user_name,
            price_drops=price_drops,
            price_increases=price_increases
        )
        
        return self.send_email(recipient_email, subject, html_content)
    
    def send_welcome_email(self, recipient_email: str, user_name: str) -> bool:
        """Send welcome email to newly registered users"""
        subject = "🏠 Welcome to SomonGPT Real Estate Intelligence!"
        
        # HTML template for welcome email
        html_template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Welcome to SomonGPT</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: linear-gradient(135deg, #2563eb, #10b981); color: white; padding: 30px; border-radius: 8px; text-align: center; }
                .content { padding: 20px 0; }
                .feature-card { border: 1px solid #e0e0e0; border-radius: 8px; margin: 15px 0; padding: 20px; background: #f9f9f9; }
                .feature-icon { font-size: 24px; margin-bottom: 10px; }
                .cta-button { background: #2563eb; color: white; padding: 15px 30px; text-decoration: none; border-radius: 6px; display: inline-block; margin: 20px 0; font-weight: bold; }
                .cta-button:hover { background: #1d4ed8; }
                .footer { margin-top: 30px; padding: 20px; background: #f5f5f5; border-radius: 8px; text-align: center; font-size: 14px; color: #666; }
                .highlight { background: #fef3c7; padding: 15px; border-radius: 6px; border-left: 4px solid #f59e0b; margin: 20px 0; }
                .step { background: white; border-radius: 8px; padding: 15px; margin: 10px 0; border-left: 4px solid #10b981; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎉 Welcome to SomonGPT!</h1>
                    <h2>Real Estate Intelligence Platform</h2>
                    <p>Hello {{ user_name }}, thank you for joining our AI-powered real estate investment platform!</p>
                </div>
                
                <div class="content">
                    <div class="highlight">
                        <strong>🚀 Get Started in 3 Easy Steps:</strong>
                    </div>
                    
                    <div class="step">
                        <strong>1. 📊 Collect Property Data</strong><br>
                        Start by collecting property listings based on your investment criteria. This is the foundation for all our AI-powered features.
                    </div>
                    
                    <div class="step">
                        <strong>2. 🎯 Find Bargain Properties</strong><br>
                        Our AI will analyze market data to identify undervalued properties with high investment potential.
                    </div>
                    
                    <div class="step">
                        <strong>3. 📈 Track Your Investments</strong><br>
                        Monitor price changes, receive daily market summaries, and get personalized investment alerts.
                    </div>
                    
                    <div style="text-align: center;">
                        <a href="https://somongpt.vercel.app/collect" class="cta-button">🚀 Start Collecting Data</a>
                    </div>
                    
                    <h2>🔥 Platform Features</h2>
                    
                    <div class="feature-card">
                        <div class="feature-icon">🎯</div>
                        <h3>Bargain Finder</h3>
                        <p>AI-powered analysis to identify undervalued properties with exceptional investment potential. Get alerts when new bargains are discovered.</p>
                    </div>
                    
                    <div class="feature-card">
                        <div class="feature-icon">🔍</div>
                        <h3>Smart Property Search</h3>
                        <p>Advanced filtering system to find properties matching your exact investment criteria. Search by price, location, size, and investment potential.</p>
                    </div>
                    
                    <div class="feature-card">
                        <div class="feature-icon">📊</div>
                        <h3>Market Dashboard</h3>
                        <p>Real-time market analytics with interactive charts showing price trends, investment opportunities, and market insights.</p>
                    </div>
                    
                    <div class="feature-card">
                        <div class="feature-icon">🤖</div>
                        <h3>AI Price Predictor</h3>
                        <p>Machine learning-powered property valuation tool. Get accurate price predictions based on market data and property characteristics.</p>
                    </div>
                    
                    <div class="feature-card">
                        <div class="feature-icon">📧</div>
                        <h3>Automated Notifications</h3>
                        <p>Daily market summaries, bargain alerts, and price change notifications delivered directly to your inbox.</p>
                    </div>
                    
                    <div class="highlight">
                        <strong>💡 Pro Tip:</strong> Start with data collection to unlock all features. The more data you collect, the better our AI recommendations become!
                    </div>
                    
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="https://somongpt.vercel.app/home" class="cta-button">🏠 Go to Dashboard</a>
                        <a href="https://somongpt.vercel.app/collect" class="cta-button">📊 Collect Data</a>
                    </div>
                </div>
                
                <div class="footer">
                    <p><strong>SomonGPT Real Estate Intelligence</strong></p>
                    <p>AI-Powered Property Investment Platform for Tajikistan</p>
                    <p>Need help? Contact us or visit our platform for tutorials and support.</p>
                    <p style="font-size: 12px; margin-top: 15px;">
                        This email was sent because you registered for SomonGPT Real Estate Intelligence Platform.<br>
                        If you didn't register, please ignore this email.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """)
        
        # Plain text version
        text_content = f"""
        Welcome to SomonGPT Real Estate Intelligence!
        
        Hello {user_name},
        
        Thank you for joining our AI-powered real estate investment platform!
        
        GET STARTED IN 3 EASY STEPS:
        1. Collect Property Data - Start by collecting property listings based on your criteria
        2. Find Bargain Properties - Our AI will identify undervalued properties for you
        3. Track Your Investments - Monitor prices and receive personalized alerts
        
        PLATFORM FEATURES:
        🎯 Bargain Finder - AI-powered undervalued property detection
        🔍 Smart Property Search - Advanced filtering and search capabilities
        📊 Market Dashboard - Real-time analytics and market insights
        🤖 AI Price Predictor - Machine learning property valuations
        📧 Automated Notifications - Daily summaries and investment alerts
        
                 Start your real estate investment journey today!
         Visit: https://somongpt.vercel.app/collect
        
        Best regards,
        SomonGPT Real Estate Intelligence Team
        """
        
        html_content = html_template.render(
            user_name=user_name,
        )
        
        return self.send_email(recipient_email, subject, html_content, text_content)

    def send_daily_market_summary(self, recipient_email: str, user_name: str, market_summary: Dict[str, Any]) -> bool:
        """Send daily market summary"""
        subject = f"📊 Daily Market Summary - {datetime.now().strftime('%B %d, %Y')}"
        
        html_template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Daily Market Summary</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; padding: 20px; border-radius: 8px; text-align: center; }
                .stat-card { background: #f8fafc; padding: 15px; border-radius: 8px; margin: 10px 0; }
                .stat-value { font-size: 24px; font-weight: bold; color: #2563eb; }
                .footer { margin-top: 30px; padding: 20px; background: #f5f5f5; border-radius: 8px; text-align: center; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Daily Market Summary</h1>
                    <p>{{ date_str }}</p>
                    <p>Hello {{ user_name }}, here's your daily market update!</p>
                </div>
                
                <div style="margin: 20px 0;">
                    <div class="stat-card">
                        <div class="stat-value">{{ new_listings }}</div>
                        <div>New Listings Added</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-value">{{ new_bargains }}</div>
                        <div>New Bargain Properties</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-value">{{ price_changes }}</div>
                        <div>Price Changes in Favorites</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-value">{{ "{:,.0f}".format(avg_price) }} TJS</div>
                        <div>Average Market Price</div>
                    </div>
                </div>
                
                <div class="footer">
                    <p>Stay ahead of the market with SomonGPT!</p>
                </div>
            </div>
        </body>
        </html>
        """)
        
        html_content = html_template.render(
            user_name=user_name,
            date_str=datetime.now().strftime('%B %d, %Y'),
            **market_summary
        )
        
        return self.send_email(recipient_email, subject, html_content)

# Global instance
notification_service = EmailNotificationService()