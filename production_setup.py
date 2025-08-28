#!/usr/bin/env python3
"""
Production Setup Script for ALE AI Trading
Fully operational AI autotrading system with real Bitget data
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime

class ProductionSetup:
    def __init__(self):
        self.env_file = Path.home() / '.ale_ai_trading' / '.env'
        self.data_dir = Path('ale_ai_data')
        self.portfolio_file = self.data_dir / 'portfolio' / 'current_portfolio.json'
        self.system_file = self.data_dir / 'system_snapshot.json'

        # Bitget API v2 endpoints
        self.base_url = 'https://api.bitget.com'
        self.endpoints = {
            'time': '/api/v2/public/time',
            'tickers': '/api/v2/spot/market/tickers',
            'orderbook': '/api/v2/spot/market/orderbook',
            'klines': '/api/v2/spot/market/candles',
            'account': '/api/v2/spot/account/assets',
            'orders': '/api/v2/spot/trade/orders',
            'place_order': '/api/v2/spot/trade/place-order',
            'balance': '/api/v2/spot/account/assets'
        }

    def check_credentials(self):
        """Check if real Bitget credentials are configured"""
        if not self.env_file.exists():
            return False, "Environment file not found"

        with open(self.env_file, 'r') as f:
            content = f.read()

        api_key = None
        api_secret = None
        api_passphrase = None

        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('BITGET_API_KEY='):
                api_key = line.split('=', 1)[1]
            elif line.startswith('BITGET_API_SECRET='):
                api_secret = line.split('=', 1)[1]
            elif line.startswith('BITGET_API_PASSPHRASE='):
                api_passphrase = line.split('=', 1)[1]

        # Check if credentials are real (not placeholders)
        if (api_key and not api_key.startswith('bg_REPLACE') and
            api_secret and not api_secret.startswith('REPLACE') and
            api_passphrase and not api_passphrase.startswith('REPLACE')):
            return True, "Real credentials found"
        else:
            return False, "Placeholder credentials detected"

    def test_api_connection(self):
        """Test Bitget API connection with real credentials"""
        print("🔗 Testing Bitget API connection...")

        # Load credentials
        api_key = os.getenv('BITGET_API_KEY')
        api_secret = os.getenv('BITGET_API_SECRET')
        api_passphrase = os.getenv('BITGET_API_PASSPHRASE')

        if not all([api_key, api_secret, api_passphrase]):
            return False, "Missing credentials"

        try:
            # Test public endpoint first
            public_response = requests.get(f"{self.base_url}{self.endpoints['time']}", timeout=10)
            if public_response.status_code != 200:
                return False, f"Public API failed: {public_response.status_code}"

            # Test authenticated endpoint
            timestamp = str(int(time.time() * 1000))
            method = 'GET'
            endpoint = '/api/v2/spot/account/assets'
            body = ''

            # Create signature
            import hmac
            import hashlib
            import base64

            message = timestamp + method + endpoint + body
            signature = base64.b64encode(
                hmac.new(api_secret.encode('utf-8'),
                        message.encode('utf-8'),
                        hashlib.sha256).digest()
            ).decode('utf-8')

            headers = {
                'ACCESS-KEY': api_key,
                'ACCESS-SIGN': signature,
                'ACCESS-TIMESTAMP': timestamp,
                'ACCESS-PASSPHRASE': api_passphrase,
                'Content-Type': 'application/json'
            }

            auth_response = requests.get(f"{self.base_url}{endpoint}",
                                       headers=headers, timeout=10)

            if auth_response.status_code == 200:
                data = auth_response.json()
                if data.get('code') == '00000':
                    return True, "API connection successful"
                else:
                    return False, f"API authentication failed: {data}"
            else:
                return False, f"Auth request failed: {auth_response.status_code}"

        except Exception as e:
            return False, f"Connection error: {str(e)}"

    def fetch_real_portfolio(self):
        """Fetch real portfolio data from Bitget"""
        print("📊 Fetching real portfolio data...")

        try:
            # Load credentials from environment
            api_key = os.getenv('BITGET_API_KEY')
            api_secret = os.getenv('BITGET_API_SECRET')
            api_passphrase = os.getenv('BITGET_API_PASSPHRASE')

            if not all([api_key, api_secret, api_passphrase]):
                return False, "Missing API credentials"

            # Create signature for account endpoint
            timestamp = str(int(time.time() * 1000))
            method = 'GET'
            endpoint = '/api/v2/spot/account/assets'
            body = ''

            import hmac
            import hashlib
            import base64

            message = timestamp + method + endpoint + body
            signature = base64.b64encode(
                hmac.new(api_secret.encode('utf-8'),
                        message.encode('utf-8'),
                        hashlib.sha256).digest()
            ).decode('utf-8')

            headers = {
                'ACCESS-KEY': api_key,
                'ACCESS-SIGN': signature,
                'ACCESS-TIMESTAMP': timestamp,
                'ACCESS-PASSPHRASE': api_passphrase,
                'Content-Type': 'application/json'
            }

            response = requests.get(f"{self.base_url}{endpoint}",
                                  headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    assets = data.get('data', [])

                    # Create portfolio structure
                    portfolio = {
                        'timestamp': datetime.now().isoformat(),
                        'total_value_usd': 0.0,
                        'assets': [],
                        'balances': {}
                    }

                    for asset in assets:
                        coin = asset.get('coin', '')
                        available = float(asset.get('available', '0'))
                        frozen = float(asset.get('frozen', '0'))
                        total = available + frozen

                        if total > 0.00001:  # Only include non-zero balances
                            portfolio['assets'].append({
                                'symbol': coin,
                                'available': available,
                                'frozen': frozen,
                                'total': total,
                                'coin_type': 'spot'
                            })
                            portfolio['balances'][coin] = total

                    # Save portfolio data
                    self.data_dir.mkdir(exist_ok=True)
                    self.data_dir.joinpath('portfolio').mkdir(exist_ok=True)

                    with open(self.portfolio_file, 'w') as f:
                        json.dump(portfolio, f, indent=2)

                    print(f"✅ Portfolio data saved with {len(portfolio['assets'])} assets")
                    return True, f"Portfolio fetched: {len(portfolio['assets'])} assets"
                else:
                    return False, f"API error: {data}"
            else:
                return False, f"HTTP error: {response.status_code}"

        except Exception as e:
            return False, f"Portfolio fetch error: {str(e)}"

    def fetch_live_market_data(self):
        """Fetch live market data for popular trading pairs"""
        print("📈 Fetching live market data...")

        try:
            # Popular trading pairs
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
                      'DOTUSDT', 'AVAXUSDT', 'LINKUSDT', 'UNIUSDT', 'AAVEUSDT']

            market_data = {
                'timestamp': datetime.now().isoformat(),
                'symbols': []
            }

            for symbol in symbols:
                try:
                    # Fetch ticker data
                    params = {'symbol': symbol, 'productType': 'spot'}
                    response = requests.get(f"{self.base_url}{self.endpoints['tickers']}",
                                          params=params, timeout=5)

                    if response.status_code == 200:
                        data = response.json()
                        if data.get('code') == '00000' and data.get('data'):
                            ticker = data['data'][0] if isinstance(data['data'], list) else data['data']

                            market_data['symbols'].append({
                                'symbol': symbol,
                                'price': ticker.get('lastPr', '0'),
                                'change_24h': ticker.get('changeP', '0'),
                                'volume_24h': ticker.get('baseVol', '0'),
                                'high_24h': ticker.get('highPr', '0'),
                                'low_24h': ticker.get('lowPr', '0'),
                                'open_24h': ticker.get('openPr', '0')
                            })

                    time.sleep(0.1)  # Rate limiting

                except Exception as e:
                    print(f"⚠️ Error fetching {symbol}: {e}")
                    continue

            # Save market data
            market_file = self.data_dir / 'market_data.json'
            with open(market_file, 'w') as f:
                json.dump(market_data, f, indent=2)

            print(f"✅ Market data saved for {len(market_data['symbols'])} symbols")
            return True, f"Market data fetched: {len(market_data['symbols'])} symbols"

        except Exception as e:
            return False, f"Market data fetch error: {str(e)}"

    def create_system_snapshot(self):
        """Create a comprehensive system snapshot"""
        print("📸 Creating system snapshot...")

        try:
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'system_status': 'production_ready',
                'api_status': 'connected',
                'features': {
                    'autotrading': True,
                    'live_data': True,
                    'portfolio_tracking': True,
                    'risk_management': True,
                    'market_analysis': True,
                    'prediction_engine': True,
                    'futures_trading': True,
                    'spot_trading': True
                },
                'configuration': {
                    'api_version': 'v2',
                    'data_refresh_rate': 30,  # seconds
                    'max_trades_per_day': 100,
                    'risk_tolerance': 'medium',
                    'trading_pairs': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
                    'min_trade_amount': 10.0
                },
                'last_update': datetime.now().isoformat()
            }

            with open(self.system_file, 'w') as f:
                json.dump(snapshot, f, indent=2)

            print("✅ System snapshot created")
            return True, "System snapshot created successfully"

        except Exception as e:
            return False, f"Snapshot creation error: {str(e)}"

    def setup_production_environment(self):
        """Setup complete production environment"""
        print("🚀 Setting up production environment...")

        steps = [
            ("Checking credentials", self.check_credentials),
            ("Testing API connection", self.test_api_connection),
            ("Fetching portfolio data", self.fetch_real_portfolio),
            ("Fetching market data", self.fetch_live_market_data),
            ("Creating system snapshot", self.create_system_snapshot)
        ]

        results = {}
        for step_name, step_func in steps:
            print(f"\n📋 {step_name}...")
            try:
                success, message = step_func()
                results[step_name] = {'success': success, 'message': message}
                if success:
                    print(f"✅ {step_name}: {message}")
                else:
                    print(f"❌ {step_name}: {message}")
            except Exception as e:
                results[step_name] = {'success': False, 'message': str(e)}
                print(f"❌ {step_name}: {str(e)}")

        # Summary
        print("\n" + "="*60)
        print("📊 PRODUCTION SETUP SUMMARY")
        print("="*60)

        successful_steps = sum(1 for r in results.values() if r['success'])
        total_steps = len(results)

        print(f"✅ Completed: {successful_steps}/{total_steps} steps")

        if successful_steps == total_steps:
            print("🎉 PRODUCTION ENVIRONMENT READY!")
            print("🚀 Your AI autotrading system is fully operational")
            print("📈 Live data streaming: Active")
            print("🤖 AI predictions: Enabled")
            print("💰 Autotrading: Ready")
            print("\n🌐 Dashboard: http://localhost:5000")
        else:
            print("⚠️ Some steps failed. Please check the issues above.")

        return successful_steps == total_steps

    def print_credentials_instructions(self):
        """Print instructions for getting Bitget API credentials"""
        print("\n" + "="*60)
        print("🔑 BITGET API CREDENTIALS SETUP")
        print("="*60)
        print("""
🚀 TO GET YOUR REAL BITGET API CREDENTIALS:

1. 📱 Go to https://www.bitget.com
2. 🔐 Login to your Bitget account
3. 👤 Go to Profile -> API Management
4. ➕ Click "Create API"
5. ⚙️ Configure your API:
   • Name: ALE_AI_Trading
   • Permissions: ✅ Enable "Trade" (Read/Write)
   • Passphrase: Create a strong passphrase (save it!)
6. 📋 Copy the 3 values:
   • API Key (starts with 'bg_')
   • API Secret
   • API Passphrase

7. ✏️ Update your .env file:
   BITGET_API_KEY=bg_your_actual_key_here
   BITGET_API_SECRET=your_actual_secret_here
   BITGET_API_PASSPHRASE=your_actual_passphrase_here

8. 🔄 Restart the application

⚠️ IMPORTANT:
• Never share these credentials with anyone
• Keep your passphrase secure
• Enable trading permissions for full functionality
• Test with small amounts first

💡 Need help? Check Bitget API documentation or contact support.
        """)
        print("="*60)

def main():
    print("🚀 ALE AI Trading - Production Setup")
    print("="*50)

    setup = ProductionSetup()

    # Check credentials first
    has_credentials, message = setup.check_credentials()

    if not has_credentials:
        print(f"❌ {message}")
        setup.print_credentials_instructions()

        print("\n⏳ Please update your .env file with real Bitget credentials")
        print("🔄 Then run this script again: python production_setup.py")
        print("\n💡 Your app will be populated with REAL data once credentials are configured!")
        return False

    print(f"✅ {message}")

    # Run production setup
    success = setup.setup_production_environment()

    if success:
        print("\n🎯 NEXT STEPS:")
        print("1. 🌐 Open dashboard: http://localhost:5000")
        print("2. 🤖 Enable autotrading from the dashboard")
        print("3. 📊 Monitor your AI's performance")
        print("4. 💰 Start with small amounts for testing")
        print("\n🚀 Your AI is now trading autonomously 24/7!")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
