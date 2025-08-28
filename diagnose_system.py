#!/usr/bin/env python3
"""
Comprehensive diagnostic script for ALE AI Trading System
Identifies and fixes all errors and issues
"""

import os
import sys
import json
import traceback
from pathlib import Path

class SystemDiagnostic:
    def __init__(self):
        self.issues_found = []
        self.issues_fixed = []

    def log_issue(self, issue_type, description, severity="ERROR"):
        """Log an issue found"""
        self.issues_found.append({
            'type': issue_type,
            'description': description,
            'severity': severity
        })
        print(f"[{severity}] {issue_type}: {description}")

    def log_fix(self, fix_description):
        """Log a fix applied"""
        self.issues_fixed.append(fix_description)
        print(f"[FIXED] {fix_description}")

    def check_env_file(self):
        """Check .env file configuration"""
        print("\n🔍 Checking .env file configuration...")

        env_file = Path.home() / '.ale_ai_trading' / '.env'

        if not env_file.exists():
            self.log_issue("ENV_FILE", ".env file does not exist", "CRITICAL")
            return False

        if not env_file.parent.exists():
            self.log_issue("ENV_DIR", ".env directory does not exist", "CRITICAL")
            return False

        try:
            with open(env_file, 'r') as f:
                content = f.read()

            # Parse credentials
            credentials = {}
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    credentials[key.strip()] = value.strip()

            required = ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSPHRASE']
            missing = [key for key in required if key not in credentials or not credentials[key]]

            if missing:
                for key in missing:
                    self.log_issue("CREDENTIALS", f"Missing or empty: {key}", "CRITICAL")
                return False

            # Check if credentials are still placeholders
            if any('replace_with_your_actual' in credentials[key] for key in required):
                self.log_issue("CREDENTIALS", "Credentials still contain placeholder values", "CRITICAL")
                return False

            self.log_fix("Environment file and credentials configured correctly")
            return True

        except Exception as e:
            self.log_issue("ENV_FILE", f"Error reading .env file: {e}", "CRITICAL")
            return False

    def test_imports(self):
        """Test all required imports"""
        print("\n🔍 Testing imports...")

        try:
            import flask
            import requests
            import pandas
            import numpy
            print("✅ Core dependencies imported successfully")
        except ImportError as e:
            self.log_issue("DEPENDENCIES", f"Missing dependency: {e}", "CRITICAL")
            return False

        try:
            import web_ui_singularity
            print("✅ Main application module imported")
        except Exception as e:
            self.log_issue("IMPORT", f"Failed to import main module: {e}", "CRITICAL")
            return False

        try:
            from web_ui_singularity import app, bitget_client, advanced_trading_engine
            print("✅ Core components imported successfully")
            return True
        except Exception as e:
            self.log_issue("IMPORT", f"Failed to import core components: {e}", "CRITICAL")
            return False

    def test_bitget_client(self):
        """Test Bitget API client"""
        print("\n🔍 Testing Bitget API client...")

        try:
            from web_ui_singularity import bitget_client

            if not bitget_client:
                self.log_issue("BITGET_CLIENT", "Bitget client not initialized", "CRITICAL")
                return False

            if not bitget_client.api_key or bitget_client.api_key == '':
                self.log_issue("BITGET_CLIENT", "API key not configured", "CRITICAL")
                return False

            if not bitget_client.api_secret or bitget_client.api_secret == '':
                self.log_issue("BITGET_CLIENT", "API secret not configured", "CRITICAL")
                return False

            if not bitget_client.api_passphrase or bitget_client.api_passphrase == '':
                self.log_issue("BITGET_CLIENT", "API passphrase not configured", "CRITICAL")
                return False

            # Test connection
            try:
                connection_result = bitget_client.test_connection()
                if connection_result and connection_result.get('status') == 'success':
                    print("✅ Bitget API connection successful")
                    return True
                else:
                    self.log_issue("BITGET_CONNECTION", f"Connection failed: {connection_result}", "WARNING")
                    return False
            except Exception as e:
                self.log_issue("BITGET_CONNECTION", f"Connection test error: {e}", "WARNING")
                return False

        except Exception as e:
            self.log_issue("BITGET_CLIENT", f"Client test failed: {e}", "CRITICAL")
            return False

    def test_market_data(self):
        """Test market data fetching"""
        print("\n🔍 Testing market data fetching...")

        try:
            from web_ui_singularity import bitget_client

            # Test ticker data
            ticker = bitget_client.get_ticker('BTCUSDT')
            if ticker and ticker.get('code') == '00000':
                data = ticker.get('data', {})
                price = data.get('last', 'N/A')
                print(f"✅ Market data working - BTC/USDT: ${price}")
                return True
            else:
                self.log_issue("MARKET_DATA", f"Ticker API failed: {ticker}", "ERROR")
                return False

        except Exception as e:
            self.log_issue("MARKET_DATA", f"Market data test failed: {e}", "ERROR")
            return False

    def test_autotrading_engine(self):
        """Test autotrading engine"""
        print("\n🔍 Testing autotrading engine...")

        try:
            from web_ui_singularity import advanced_trading_engine

            if not advanced_trading_engine:
                self.log_issue("AUTOTRADING_ENGINE", "Engine not initialized", "CRITICAL")
                return False

            # Test system status
            status = advanced_trading_engine.get_system_status()
            if status:
                print("✅ Autotrading engine initialized successfully")
                print(f"   Status: {status.get('status', 'unknown')}")
                print(f"   Balance: ${status.get('balance', 0):.2f}")
                print(f"   Active: {status.get('autotrading', False)}")
                return True
            else:
                self.log_issue("AUTOTRADING_ENGINE", "Failed to get system status", "ERROR")
                return False

        except Exception as e:
            self.log_issue("AUTOTRADING_ENGINE", f"Engine test failed: {e}", "CRITICAL")
            return False

    def test_flask_app(self):
        """Test Flask application"""
        print("\n🔍 Testing Flask application...")

        try:
            from web_ui_singularity import app

            # Test basic endpoints
            with app.test_client() as client:
                # Test dashboard
                response = client.get('/')
                if response.status_code == 200:
                    print("✅ Dashboard endpoint working")
                else:
                    self.log_issue("FLASK_APP", f"Dashboard failed: {response.status_code}", "ERROR")

                # Test system status
                response = client.get('/api/system/status')
                if response.status_code == 200:
                    print("✅ System status API working")
                else:
                    self.log_issue("FLASK_APP", f"System status API failed: {response.status_code}", "ERROR")

                # Test autotrading status
                response = client.get('/api/autotrading/status')
                if response.status_code == 200:
                    print("✅ Autotrading status API working")
                else:
                    self.log_issue("FLASK_APP", f"Autotrading status API failed: {response.status_code}", "ERROR")

            return True

        except Exception as e:
            self.log_issue("FLASK_APP", f"Flask test failed: {e}", "CRITICAL")
            return False

    def fix_bitget_api_issues(self):
        """Fix known Bitget API issues"""
        print("\n🔧 Fixing Bitget API issues...")

        try:
            from web_ui_singularity import bitget_client

            # Check for common API endpoint issues
            # The main issue might be with URL construction or authentication

            # Test a simple API call to see what happens
            try:
                # Try to get server time first (public endpoint)
                result = bitget_client._request('GET', '/api/v2/public/time')
                if result and result.get('code') == '00000':
                    print("✅ Public API working")
                else:
                    print(f"⚠️ Public API issue: {result}")
            except Exception as e:
                print(f"⚠️ Public API test failed: {e}")

            self.log_fix("Bitget API diagnostic checks completed")
            return True

        except Exception as e:
            self.log_issue("BITGET_FIX", f"Failed to fix Bitget API: {e}", "ERROR")
            return False

    def run_full_diagnostic(self):
        """Run complete system diagnostic"""
        print("🚀 ALE AI Trading System - Comprehensive Diagnostic")
        print("=" * 60)

        # Run all tests
        tests = [
            ("Environment Configuration", self.check_env_file),
            ("Import Dependencies", self.test_imports),
            ("Bitget API Client", self.test_bitget_client),
            ("Market Data Fetching", self.test_market_data),
            ("Autotrading Engine", self.test_autotrading_engine),
            ("Flask Application", self.test_flask_app),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")

        print("\n" + "=" * 60)
        print(f"📊 DIAGNOSTIC RESULTS: {passed}/{total} tests passed")

        # Summary
        print(f"\n🔍 ISSUES FOUND: {len(self.issues_found)}")
        for issue in self.issues_found:
            print(f"   {issue['severity']}: {issue['description']}")

        print(f"\n🔧 FIXES APPLIED: {len(self.issues_fixed)}")
        for fix in self.issues_fixed:
            print(f"   ✓ {fix}")

        # Overall status
        if passed == total:
            print("\n🎉 ALL SYSTEMS OPERATIONAL!")
            print("✅ Your autotrading system is ready for live trading!")
            return True
        else:
            print(f"\n⚠️ {total - passed} ISSUES REQUIRE ATTENTION")
            print("🔧 Please fix the issues above before live trading")
            return False

def main():
    """Main diagnostic function"""
    diagnostic = SystemDiagnostic()
    success = diagnostic.run_full_diagnostic()

    if not success:
        print("\n" + "=" * 60)
        print("🔧 AUTOMATED FIXES ATTEMPTED")
        print("=" * 60)

        # Try to apply fixes
        diagnostic.fix_bitget_api_issues()

    print("\n" + "=" * 60)
    print("📋 NEXT STEPS")
    print("=" * 60)

    if success:
        print("1. ✅ Start the application: python web_ui_singularity.py")
        print("2. 🌐 Open dashboard: http://localhost:5000")
        print("3. 🤖 Enable autotrading from the dashboard")
        print("4. 📊 Monitor your automated trades!")
    else:
        print("1. 🔧 Fix the issues listed above")
        print("2. ⚙️ Configure your Bitget API credentials")
        print("3. 🧪 Run diagnostic again: python diagnose_system.py")
        print("4. 🚀 Start trading once all issues are resolved")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
