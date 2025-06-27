#!/usr/bin/env python3
"""
One-Time Email Configuration for SomonGPT

Configure your email credentials once and the system will use them automatically
to send notifications to all registered users.
"""

import os
import sys
import json
import getpass
from cryptography.fernet import Fernet

def generate_key():
    """Generate a key for encryption"""
    return Fernet.generate_key()

def encrypt_password(password: str, key: bytes) -> str:
    """Encrypt password"""
    fernet = Fernet(key)
    encrypted = fernet.encrypt(password.encode())
    return encrypted.decode()

def decrypt_password(encrypted_password: str, key: bytes) -> str:
    """Decrypt password"""
    fernet = Fernet(key)
    decrypted = fernet.decrypt(encrypted_password.encode())
    return decrypted.decode()

def save_email_config(email: str, password: str):
    """Save email configuration securely"""
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    os.makedirs(config_dir, exist_ok=True)
    
    # Generate encryption key
    key = generate_key()
    
    # Encrypt password
    encrypted_password = encrypt_password(password, key)
    
    # Save config
    config = {
        'email_user': email,
        'email_password': encrypted_password,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'from_name': 'SomonGPT Real Estate Intelligence'
    }
    
    config_file = os.path.join(config_dir, 'email_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save key separately
    key_file = os.path.join(config_dir, 'email_key.key')
    with open(key_file, 'wb') as f:
        f.write(key)
    
    # Make files readable only by owner
    os.chmod(config_file, 0o600)
    os.chmod(key_file, 0o600)
    
    return config_file, key_file

def load_email_config():
    """Load email configuration"""
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    config_file = os.path.join(config_dir, 'email_config.json')
    key_file = os.path.join(config_dir, 'email_key.key')
    
    if not os.path.exists(config_file) or not os.path.exists(key_file):
        return None
    
    # Load key
    with open(key_file, 'rb') as f:
        key = f.read()
    
    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Decrypt password
    config['email_password'] = decrypt_password(config['email_password'], key)
    
    return config

def test_email_config(config):
    """Test email configuration"""
    import smtplib
    import ssl
    
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            server.starttls(context=context)
            server.login(config['email_user'], config['email_password'])
        return True
    except Exception as e:
        print(f"❌ Email test failed: {str(e)}")
        return False

def main():
    print("📧 SomonGPT Email Configuration")
    print("=" * 50)
    print("Configure your email credentials once - the system will use them")
    print("to automatically send notifications to all registered users.\n")
    
    # Check if config already exists
    existing_config = load_email_config()
    if existing_config:
        print(f"✅ Email configuration found for: {existing_config['email_user']}")
        print("\nOptions:")
        print("1. Test current configuration")
        print("2. Update configuration")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\n🧪 Testing email configuration...")
            if test_email_config(existing_config):
                print("✅ Email configuration is working!")
            return
        elif choice == "3":
            return
        # choice == "2" continues to reconfigure
    
    print("\n📋 Email Setup Instructions:")
    print("1. This will be YOUR email that SENDS notifications")
    print("2. Notifications will be sent TO users who registered on your platform")
    print("3. You need a Gmail App Password (not your regular password)")
    print("\n💡 Gmail App Password Setup:")
    print("   • Go to: myaccount.google.com → Security")
    print("   • Enable 2-Step Verification (if not already)")
    print("   • App passwords → Generate for 'Mail'")
    print("   • Use the 16-character password (e.g., 'abcd efgh ijkl mnop')")
    
    print("\n" + "="*50)
    
    # Get email credentials
    email = input("Enter your Gmail address (sender): ").strip()
    if not email:
        print("❌ Email address required")
        return
    
    print(f"\nEnter App Password for {email}")
    print("(This will be hidden as you type)")
    password = getpass.getpass("App Password: ").strip()
    if not password:
        print("❌ App Password required")
        return
    
    # Test configuration
    print("\n🧪 Testing email configuration...")
    test_config = {
        'email_user': email,
        'email_password': password,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587
    }
    
    if not test_email_config(test_config):
        print("❌ Email test failed. Please check your credentials.")
        return
    
    print("✅ Email test successful!")
    
    # Save configuration
    print("\n💾 Saving configuration...")
    config_file, key_file = save_email_config(email, password)
    
    print(f"✅ Email configuration saved!")
    print(f"   📄 Config: {config_file}")
    print(f"   🔑 Key: {key_file}")
    
    print(f"\n🎯 How it works now:")
    print(f"   📧 Sender: {email} (your email)")
    print(f"   👥 Recipients: 4 registered users with notifications enabled")
    print(f"   ⚡ Automatic: System sends notifications without asking for credentials")
    
    print(f"\n🚀 To start automated notifications:")
    print(f"   python scripts/enhanced_daily_scheduler.py")
    
    print(f"\n🧪 To test with real emails:")
    print(f"   python scripts/test_notifications_production.py")

if __name__ == "__main__":
    main() 