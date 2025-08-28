#!/usr/bin/env python3
"""
ALE AI Trading System - Configuration Management
Secure API credential management for Bitget integration
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional
import getpass

class ConfigManager:
    """Secure configuration management for API credentials"""
    
    def __init__(self):
        self.config_dir = Path.home() / '.ale_ai_trading'
        self.config_file = self.config_dir / 'config.json'
        self.credentials_file = self.config_dir / 'credentials.json'
        
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(exist_ok=True)
        
        # Set secure file permissions
        if self.config_dir.exists():
            os.chmod(self.config_dir, 0o700)
    
    def setup_credentials(self) -> bool:
        """Interactive setup of API credentials"""
        print("🔐 Setting up Bitget API Credentials")
        print("=" * 50)
        
        try:
            # Get API credentials from user
            api_key = getpass.getpass("Enter your Bitget API Key: ")
            api_secret = getpass.getpass("Enter your Bitget API Secret: ")
            api_passphrase = getpass.getpass("Enter your Bitget API Passphrase: ")
            
            # Validate credentials
            if not all([api_key, api_secret, api_passphrase]):
                print("❌ All credentials are required!")
                return False
            
            # Test credentials (optional - can be skipped)
            test_connection = input("Test connection now? (y/n): ").lower().strip()
            
            if test_connection == 'y':
                if self._test_credentials(api_key, api_secret, api_passphrase):
                    print("✅ API credentials validated successfully!")
                else:
                    print("❌ API credentials validation failed!")
                    retry = input("Continue anyway? (y/n): ").lower().strip()
                    if retry != 'y':
                        return False
            
            # Save credentials securely
            credentials = {
                'bitget_api_key': api_key,
                'bitget_api_secret': api_secret,
                'bitget_api_passphrase': api_passphrase,
                'setup_date': self._get_current_timestamp()
            }
            
            self._save_credentials(credentials)
            
            # Create environment file
            self._create_env_file(credentials)
            
            print("✅ API credentials saved successfully!")
            print("🔒 Credentials are stored securely in:", self.credentials_file)
            print("🌍 Environment variables set in:", self.config_dir / '.env')
            
            return True
            
        except KeyboardInterrupt:
            print("\n❌ Setup cancelled by user")
            return False
        except Exception as e:
            print(f"❌ Error during setup: {e}")
            return False
    
    def _test_credentials(self, api_key: str, api_secret: str, api_passphrase: str) -> bool:
        """Test API credentials with a simple API call"""
        try:
            import requests
            import time
            import hashlib
            import hmac
            import base64
            
            # Test endpoint
            url = "https://api.bitget.com/api/spot/v1/account/account"
            timestamp = str(int(time.time() * 1000))
            
            # Generate signature
            message = timestamp + "GET" + "/api/spot/v1/account/account"
            signature = base64.b64encode(
                hmac.new(
                    api_secret.encode('utf-8'),
                    message.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode()
            
            headers = {
                'ACCESS-KEY': api_key,
                'ACCESS-SIGN': signature,
                'ACCESS-TIMESTAMP': timestamp,
                'ACCESS-PASSPHRASE': api_passphrase,
                'Content-Type': 'application/json'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == '00000':
                    return True
                else:
                    print(f"❌ API Error: {result.get('msg', 'Unknown error')}")
                    return False
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def _save_credentials(self, credentials: Dict) -> None:
        """Save credentials to secure file"""
        try:
            # Save to credentials file
            with open(self.credentials_file, 'w') as f:
                json.dump(credentials, f, indent=2)
            
            # Set secure file permissions (readable only by owner)
            os.chmod(self.credentials_file, 0o600)
            
        except Exception as e:
            raise Exception(f"Failed to save credentials: {e}")
    
    def _create_env_file(self, credentials: Dict) -> None:
        """Create .env file with environment variables"""
        try:
            env_file = self.config_dir / '.env'
            
            env_content = f"""# ALE AI Trading System - Environment Variables
# Generated on: {self._get_current_timestamp()}
# WARNING: Keep this file secure and never share it!

BITGET_API_KEY={credentials['bitget_api_key']}
BITGET_API_SECRET={credentials['bitget_api_secret']}
BITGET_API_PASSPHRASE={credentials['bitget_api_passphrase']}

# Trading Configuration
TRADING_MODE=live  # Options: demo, paper, live
MAX_DAILY_LOSS=5   # Maximum daily loss percentage
MAX_POSITION_SIZE=10  # Maximum position size percentage
STOP_LOSS=2        # Default stop loss percentage
TAKE_PROFIT=6      # Default take profit percentage

# AI Configuration
AI_CONFIDENCE_THRESHOLD=0.75
AI_UPDATE_INTERVAL=30  # seconds
AUTO_REBALANCE=true
CONSCIOUSNESS_LEARNING=true

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=ai_trading.log
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=5
"""
            
            with open(env_file, 'w') as f:
                f.write(env_content)
            
            # Set secure file permissions
            os.chmod(env_file, 0o600)
            
        except Exception as e:
            raise Exception(f"Failed to create .env file: {e}")
    
    def load_credentials(self) -> Optional[Dict]:
        """Load saved credentials"""
        try:
            if self.credentials_file.exists():
                with open(self.credentials_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"❌ Error loading credentials: {e}")
            return None
    
    def get_env_variables(self) -> Dict[str, str]:
        """Get environment variables from .env file"""
        env_vars = {}
        env_file = self.config_dir / '.env'
        
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key.strip()] = value.strip()
            except Exception as e:
                print(f"❌ Error reading .env file: {e}")
        
        return env_vars
    
    def update_config(self, key: str, value: str) -> bool:
        """Update configuration value"""
        try:
            env_file = self.config_dir / '.env'
            
            if env_file.exists():
                # Read current content
                with open(env_file, 'r') as f:
                    lines = f.readlines()
                
                # Update or add the key-value pair
                updated = False
                for i, line in enumerate(lines):
                    if line.startswith(f"{key}="):
                        lines[i] = f"{key}={value}\n"
                        updated = True
                        break
                
                if not updated:
                    lines.append(f"{key}={value}\n")
                
                # Write back
                with open(env_file, 'w') as f:
                    f.writelines(lines)
                
                return True
            else:
                print("❌ .env file not found. Run setup first.")
                return False
                
        except Exception as e:
            print(f"❌ Error updating config: {e}")
            return False
    
    def show_config(self) -> None:
        """Display current configuration"""
        print("🔧 Current Configuration")
        print("=" * 50)
        
        credentials = self.load_credentials()
        env_vars = self.get_env_variables()
        
        if credentials:
            print("✅ API Credentials: Configured")
            print(f"   Setup Date: {credentials.get('setup_date', 'Unknown')}")
        else:
            print("❌ API Credentials: Not configured")
        
        print("\n📊 Trading Settings:")
        for key, value in env_vars.items():
            if not key.startswith('BITGET_'):
                print(f"   {key}: {value}")
    
    def reset_config(self) -> bool:
        """Reset all configuration"""
        try:
            confirm = input("⚠️ This will delete ALL configuration. Are you sure? (yes/no): ")
            if confirm.lower() == 'yes':
                if self.credentials_file.exists():
                    self.credentials_file.unlink()
                env_file = self.config_dir / '.env'
                if env_file.exists():
                    env_file.unlink()
                print("✅ Configuration reset successfully")
                return True
            else:
                print("❌ Configuration reset cancelled")
                return False
        except Exception as e:
            print(f"❌ Error resetting config: {e}")
            return False
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().isoformat()

def main():
    """Main configuration setup"""
    config_manager = ConfigManager()
    
    print("🧠 ALE AI Trading System - Configuration Setup")
    print("=" * 60)
    
    while True:
        print("\n📋 Configuration Options:")
        print("1. Setup API Credentials")
        print("2. Show Current Configuration")
        print("3. Update Configuration")
        print("4. Reset Configuration")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            if config_manager.setup_credentials():
                print("\n🎉 Setup completed successfully!")
                print("You can now run the trading system with: python web_ui_singularity.py")
            else:
                print("\n❌ Setup failed. Please try again.")
        
        elif choice == '2':
            config_manager.show_config()
        
        elif choice == '3':
            key = input("Enter configuration key: ").strip()
            value = input("Enter new value: ").strip()
            if config_manager.update_config(key, value):
                print("✅ Configuration updated successfully!")
            else:
                print("❌ Failed to update configuration")
        
        elif choice == '4':
            config_manager.reset_config()
        
        elif choice == '5':
            print("👋 Configuration setup completed. Goodbye!")
            break
        
        else:
            print("❌ Invalid option. Please select 1-5.")

if __name__ == '__main__':
    main()
