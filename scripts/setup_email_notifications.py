#!/usr/bin/env python3
"""
Email Notification Setup Script for SomonGPT Real Estate Platform

This script helps you configure email notifications for:
- New bargain property alerts
- Favorite property price change notifications
- Daily market summaries
"""

import os
import sys
import getpass
from typing import Dict, Any

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'somon_project'))

def create_env_file(config: Dict[str, str]):
    """Create or update .env file with email configuration"""
    env_file = os.path.join(project_root, '.env')
    
    # Read existing .env file if it exists
    existing_config = {}
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    existing_config[key] = value
    
    # Update with new email config
    existing_config.update(config)
    
    # Write back to .env file
    with open(env_file, 'w') as f:
        f.write("# SomonGPT Real Estate Platform Configuration\n")
        f.write("# Email Notification Settings\n\n")
        
        for key, value in existing_config.items():
            f.write(f"{key}={value}\n")
    
    print(f"✅ Configuration saved to {env_file}")

def test_email_configuration(config: Dict[str, str]) -> bool:
    """Test email configuration by sending a test email"""
    try:
        # Set environment variables temporarily
        for key, value in config.items():
            os.environ[key] = value
        
        # Import and test notification service
        from somon_project.webapp.backend.services.notification_service import EmailNotificationService
        
        service = EmailNotificationService()
        
        # Send test email
        test_email = config['SENDER_EMAIL']
        success = service.send_email(
            recipient_email=test_email,
            subject="🧪 SomonGPT Email Test",
            html_content="""
            <h2>✅ Email Configuration Test Successful!</h2>
            <p>Your SomonGPT email notifications are now configured correctly.</p>
            <p>You will receive notifications for:</p>
            <ul>
                <li>🎯 New bargain properties</li>
                <li>📈 Price changes in your favorite properties</li>
                <li>📊 Daily market summaries</li>
            </ul>
            <p>Best regards,<br>SomonGPT Real Estate Intelligence</p>
            """,
            text_content="""
            Email Configuration Test Successful!
            
            Your SomonGPT email notifications are now configured correctly.
            You will receive notifications for:
            - New bargain properties
            - Price changes in your favorite properties
            - Daily market summaries
            
            Best regards,
            SomonGPT Real Estate Intelligence
            """
        )
        
        if success:
            print("✅ Test email sent successfully!")
            print(f"📧 Check your inbox at {test_email}")
            return True
        else:
            print("❌ Test email failed to send")
            return False
            
    except Exception as e:
        print(f"❌ Email test failed: {str(e)}")
        return False

def setup_gmail_configuration():
    """Setup Gmail SMTP configuration"""
    print("\n📧 Gmail Configuration Setup")
    print("=" * 40)
    print("To use Gmail for notifications, you need:")
    print("1. A Gmail account")
    print("2. An App Password (not your regular password)")
    print("\nHow to get an App Password:")
    print("1. Go to your Google Account settings")
    print("2. Security → 2-Step Verification (enable if not already)")
    print("3. Security → App passwords")
    print("4. Generate an app password for 'Mail'")
    print("5. Use that 16-character password below")
    
    email = input("\n📨 Enter your Gmail address: ").strip()
    
    print(f"\n🔐 Enter the App Password for {email}")
    print("(This will be hidden as you type)")
    password = getpass.getpass("App Password: ").strip()
    
    sender_name = input("\n👤 Enter sender name (default: SomonGPT Real Estate): ").strip()
    if not sender_name:
        sender_name = "SomonGPT Real Estate Intelligence"
    
    config = {
        'SMTP_SERVER': 'smtp.gmail.com',
        'SMTP_PORT': '587',
        'SENDER_EMAIL': email,
        'SENDER_PASSWORD': password,
        'SENDER_NAME': sender_name
    }
    
    return config

def setup_custom_smtp():
    """Setup custom SMTP configuration"""
    print("\n📧 Custom SMTP Configuration")
    print("=" * 40)
    
    smtp_server = input("SMTP Server (e.g., smtp.gmail.com): ").strip()
    smtp_port = input("SMTP Port (default: 587): ").strip() or "587"
    email = input("Email Address: ").strip()
    password = getpass.getpass("Email Password: ").strip()
    sender_name = input("Sender Name (default: SomonGPT Real Estate): ").strip()
    
    if not sender_name:
        sender_name = "SomonGPT Real Estate Intelligence"
    
    config = {
        'SMTP_SERVER': smtp_server,
        'SMTP_PORT': smtp_port,
        'SENDER_EMAIL': email,
        'SENDER_PASSWORD': password,
        'SENDER_NAME': sender_name
    }
    
    return config

def main():
    """Main setup function"""
    print("🌟 SomonGPT Email Notification Setup")
    print("=" * 50)
    print("This script will help you configure email notifications")
    print("for bargain alerts and favorite property price changes.")
    
    while True:
        print("\n📧 Choose your email provider:")
        print("1. Gmail (recommended)")
        print("2. Custom SMTP server")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            config = setup_gmail_configuration()
            break
        elif choice == "2":
            config = setup_custom_smtp()
            break
        elif choice == "3":
            print("👋 Setup cancelled")
            return
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    # Test configuration
    print("\n🧪 Testing email configuration...")
    if test_email_configuration(config):
        # Save configuration
        create_env_file(config)
        
        print("\n🎉 Email notifications setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Users can enable email notifications in their profile settings")
        print("2. Run the daily scheduler to start automated notifications:")
        print(f"   cd {project_root}")
        print("   python somon_project/scripts/enhanced_daily_scheduler.py --run-now")
        print("3. To schedule daily runs:")
        print("   python somon_project/scripts/enhanced_daily_scheduler.py")
        
        print("\n📧 Notification Types:")
        print("• 🎯 New bargain properties (daily)")
        print("• 📈 Favorite property price changes (daily)")
        print("• 📊 Market summaries (optional)")
        
    else:
        print("\n❌ Email configuration test failed.")
        print("Please check your settings and try again.")
        
        retry = input("\nWould you like to try again? (y/n): ").strip().lower()
        if retry == 'y':
            main()

if __name__ == "__main__":
    main() 