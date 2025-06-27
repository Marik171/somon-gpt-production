#!/usr/bin/env python3
"""
Simple Real Data Notification Test
Send notifications with real property data using the existing config format
"""

import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
from cryptography.fernet import Fernet
from datetime import datetime
import os

def load_email_config():
    """Load email configuration"""
    try:
        # Load encryption key
        with open('config/email_key.key', 'rb') as key_file:
            key = key_file.read()
        
        # Load config
        with open('config/email_config.json', 'r') as config_file:
            config = json.load(config_file)
        
        # Decrypt password
        fernet = Fernet(key)
        decrypted_password = fernet.decrypt(config['email_password'].encode()).decode()
        
        return {
            'email': config['email_user'],
            'password': decrypted_password,
            'smtp_server': config.get('smtp_server', 'smtp.gmail.com'),
            'smtp_port': config.get('smtp_port', 587)
        }
    except Exception as e:
        print(f"❌ Error loading email config: {e}")
        return None

def send_real_data_notifications():
    """Send notifications with real property data"""
    print("🚀 REAL DATA NOTIFICATION TEST")
    print("=" * 50)
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Connect to database
    db_path = '../real_estate.db'
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get database stats
        cursor.execute('SELECT COUNT(*) FROM property_listings')
        total_properties = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(*) FROM property_listings 
            WHERE bargain_category IN ('excellent_bargain', 'good_bargain')
        ''')
        bargain_properties = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM users WHERE email_notifications = 1')
        notification_users = cursor.fetchone()[0]
        
        print(f"📊 Database Stats:")
        print(f"   🏠 Total properties: {total_properties}")
        print(f"   🎯 Bargain properties: {bargain_properties}")
        print(f"   👥 Users with notifications: {notification_users}")
        
        # Load email configuration
        email_config = load_email_config()
        if not email_config:
            print("❌ Email configuration not found")
            return
        
        print(f"📧 Using sender email: {email_config['email']}")
        
        # Get users with notifications enabled
        cursor.execute('SELECT id, email, full_name FROM users WHERE email_notifications = 1')
        users = cursor.fetchall()
        
        if not users:
            print("❌ No users found with email notifications enabled")
            return
        
        # Get bargain properties for notifications
        cursor.execute('''
            SELECT id, title, price, district, rooms, area, bargain_category, 
                   estimated_monthly_rent, gross_rental_yield
            FROM property_listings 
            WHERE bargain_category IN ('excellent_bargain', 'good_bargain')
            ORDER BY price DESC
            LIMIT 5
        ''')
        bargain_props = cursor.fetchall()
        
        print(f"🎯 Featuring {len(bargain_props)} bargain properties")
        
        # Setup email server
        print("📧 Connecting to email server...")
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['email'], email_config['password'])
        
        emails_sent = 0
        
        for user_id, email, full_name in users:
            try:
                print(f"   Sending to {email}...")
                
                # Create email
                msg = MIMEMultipart('alternative')
                msg['Subject'] = f"🎯 SomonGPT: {len(bargain_props)} Real Investment Opportunities - Khujand Market"
                msg['From'] = email_config['email']
                msg['To'] = email
                
                # Create HTML content with real property data
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; }}
                        .header {{ background: linear-gradient(135deg, #2c5aa0, #1e3a72); color: white; padding: 25px; text-align: center; }}
                        .content {{ padding: 25px; max-width: 800px; margin: 0 auto; }}
                        .property {{ border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 10px; background: #f9f9f9; }}
                        .bargain {{ background: linear-gradient(135deg, #e8f5e8, #d4edda); border-left: 5px solid #28a745; }}
                        .price {{ font-size: 1.3em; font-weight: bold; color: #2c5aa0; margin: 10px 0; }}
                        .yield {{ color: #17a2b8; font-weight: bold; }}
                        .stats {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                        .footer {{ background: #2c5aa0; color: white; padding: 20px; text-align: center; }}
                        .badge {{ background: #28a745; color: white; padding: 5px 10px; border-radius: 20px; font-size: 0.9em; }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>🏠 SomonGPT Real Estate Intelligence</h1>
                        <p style="font-size: 1.1em; margin: 10px 0;">Live Market Analysis - Khujand Real Estate</p>
                        <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 5px; margin-top: 15px;">
                            ⚡ Based on Real Scraped Data from Somon.tj
                        </div>
                    </div>
                    
                    <div class="content">
                        <h2>Dear {full_name or 'Real Estate Investor'},</h2>
                        
                        <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 20px 0;">
                            <strong>🎯 REAL DATA ALERT:</strong> Our AI has analyzed actual properties from Somon.tj and found 
                            <strong>{len(bargain_props)} excellent investment opportunities</strong> in the Khujand market.
                        </div>
                """
                
                for i, prop in enumerate(bargain_props, 1):
                    prop_id, title, price, district, rooms, area, category, rent, yield_pct = prop
                    
                    html_content += f"""
                        <div class="property bargain">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                <h3 style="margin: 0; color: #2c5aa0;">🏠 Real Property #{i}</h3>
                                <span class="badge">{category.replace('_', ' ').title()}</span>
                            </div>
                            
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0;">
                                <div>
                                    <p><strong>📍 Location:</strong> {district}, Khujand</p>
                                    <p><strong>🏠 Details:</strong> {rooms} rooms, {area:.0f} m²</p>
                                    <p class="price">💰 Price: {price:,.0f} TJS</p>
                                </div>
                                <div>
                                    <p class="yield">📈 Annual Yield: {yield_pct:.1f}%</p>
                                    <p><strong>🏠 Monthly Rent:</strong> {rent:,.0f} TJS</p>
                                    <p><strong>💵 Annual Income:</strong> {rent*12:,.0f} TJS</p>
                                </div>
                            </div>
                            
                            <div style="background: #d4edda; padding: 10px; border-radius: 5px; margin-top: 15px;">
                                <strong>💡 AI Analysis:</strong> This is a real property from Somon.tj classified as a 
                                <strong>{category.replace('_', ' ')}</strong> based on market comparison and rental yield analysis.
                            </div>
                        </div>
                    """
                
                html_content += f"""
                        <div class="stats">
                            <h3>📊 Real Market Analysis Summary</h3>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                                <div><strong>Properties Analyzed:</strong> {total_properties}</div>
                                <div><strong>Investment Opportunities:</strong> {bargain_properties}</div>
                                <div><strong>Analysis Date:</strong> {datetime.now().strftime('%B %d, %Y')}</div>
                                <div><strong>Data Source:</strong> Live Somon.tj</div>
                            </div>
                        </div>
                        
                        <div style="background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #ffc107;">
                            <strong>⚡ This is real market data!</strong> All properties shown are actual listings 
                            scraped from Somon.tj and analyzed by our AI system specifically for the Khujand real estate market.
                        </div>
                    </div>
                    
                    <div class="footer">
                        <h3 style="margin: 0 0 10px 0;">🤖 SomonGPT Real Estate Intelligence</h3>
                        <p style="margin: 5px 0;">Powered by AI Analysis of Live Market Data</p>
                        <p style="font-size: 0.9em; opacity: 0.8; margin: 10px 0 0 0;">
                            Automated notification based on real Somon.tj listings
                        </p>
                    </div>
                </body>
                </html>
                """
                
                # Add HTML content to email
                html_part = MIMEText(html_content, 'html')
                msg.attach(html_part)
                
                # Send email
                server.send_message(msg)
                emails_sent += 1
                print(f"   ✅ Sent to {email}")
                
            except Exception as e:
                print(f"   ❌ Error sending to {email}: {e}")
        
        server.quit()
        conn.close()
        
        print(f"\n🎉 REAL DATA NOTIFICATION TEST COMPLETE!")
        print(f"   📧 Emails sent: {emails_sent}/{len(users)}")
        print(f"   🎯 Real properties featured: {len(bargain_props)}")
        print(f"   📊 Data source: Actual Somon.tj listings")
        print(f"   ⏰ Completed at: {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    send_real_data_notifications()
