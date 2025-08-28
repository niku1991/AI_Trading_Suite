#!/usr/bin/env python3
"""
Test Bitget API connection and autotrading functionality
"""

import os
import sys
from pathlib import Path

def test_env_file():
    """Test if .env file exists and has credentials"""
    print("🔍 Checking .env file...")

    env_file = Path.home() / '.ale_ai_trading' / '.env'

    if not env_file.exists():
        print("❌ .env file not found!")
        print(f"   Expected location: {env_file}")
        return False

    # Read and parse credentials
    try:
        with open(env_file, 'r') as f:
            content = f.read()

        credentials = {}
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                credentials[key.strip()] = value.strip()

        # Check required credentials
        required = ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSPHRASE']
        missing = []

        for key in required:
            if key not in credentials or not credentials[key]:
                missing.append(key)
            elif 'replace_with_your_actual' in credentials[key]:
                missing.append(f"{key} (still placeholder)")

        if missing:
            print("❌ Missing or invalid credentials:")
            for item in missing:
                print(f"   • {item}")
            return False

        print("✅ .env file configured correctly")
        print(f"   🔑 API Key: {credentials['BITGET_API_KEY'][:10]}...")
        return True

    except Exception as e:
        print(f"❌ Error reading .env file: {e}")
        return False

def test_bitget_connection():
    """Test actual Bitget API connection"""
    print("\n🔍 Testing Bitget API connection...")

    try:
        import web_ui_singularity
        from web_ui_singularity import bitget_client

        if not bitget_client:
            print("❌ Bitget client not initialized")
            return False

        # Test public API first (no authentication needed)
        print("   Testing public API...")
        try:
            # Try to get server time
            result = bitget_client._request('GET', '/api/v2/public/time')
            if result and result.get('code') == '00000':
                print("   ✅ Public API working")
            else:
                print(f"   ⚠️ Public API response: {result}")
        except Exception as e:
            print(f"   ⚠️ Public API test failed: {e}")

        # Test authenticated API
        print("   Testing authenticated API...")
        try:
            # Test account info (requires authentication)
            account_info = bitget_client.get_spot_account_info()
            if account_info and account_info.get('code') == '00000':
                print("   ✅ Authenticated API working")
                print("   ✅ Credentials are valid")

                # Show account balance
                data = account_info.get('data', [])
                usdt_balance = 0
                for asset in data:
                    if asset.get('coin') == 'USDT':
                        usdt_balance = float(asset.get('available', 0))
                        break

                print(f"   💰 USDT Balance: ${usdt_balance:.2f}")
                return True

            else:
                error_msg = account_info.get('msg', 'Unknown error')
                print(f"   ❌ Authentication failed: {error_msg}")
                print("   💡 Check your API credentials in .env file")
                return False

        except Exception as e:
            print(f"   ❌ Authentication test failed: {e}")
            print("   💡 This usually means invalid API credentials")
            return False

    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def test_market_data():
    """Test market data fetching"""
    print("\n🔍 Testing market data...")

    try:
        from web_ui_singularity import bitget_client

        # Test BTC/USDT ticker
        ticker = bitget_client.get_ticker('BTCUSDT')

        if ticker and ticker.get('code') == '00000':
            data = ticker.get('data', {})
            price = data.get('last', 'N/A')
            volume = data.get('baseVolume', 'N/A')

            print("✅ Market data working!")
            print(f"   📈 BTC/USDT Price: ${price}")
            print(f"   📊 24h Volume: {volume}")

            return True
        else:
            print(f"❌ Market data failed: {ticker}")
            return False

    except Exception as e:
        print(f"❌ Market data test failed: {e}")
        return False

def test_autotrading_readiness():
    """Test if autotrading system is ready"""
    print("\n🔍 Testing autotrading readiness...")

    try:
        from web_ui_singularity import advanced_trading_engine

        if not advanced_trading_engine:
            print("❌ Autotrading engine not initialized")
            return False

        # Test system status
        status = advanced_trading_engine.get_system_status()

        if status:
            print("✅ Autotrading engine ready!")
            print(f"   📊 Status: {status.get('status', 'unknown')}")
            print(f"   💰 Balance: ${status.get('balance', 0):.2f}")
            print(f"   🎯 Capabilities: {', '.join(status.get('capabilities', []))}")

            # Check minimum balance
            balance = status.get('balance', 0)
            if balance >= 5:
                print("   ✅ Sufficient balance for trading")
                return True
            else:
                print(f"   ⚠️ Low balance: ${balance:.2f} (minimum 5 USDT needed)")
                print("   💡 Add more USDT to your Bitget account")
                return False
        else:
            print("❌ Could not get system status")
            return False

    except Exception as e:
        print(f"❌ Autotrading test failed: {e}")
        return False

def provide_troubleshooting():
    """Provide troubleshooting information"""
    print("\n" + "="*60)
    print("🔧 TROUBLESHOOTING GUIDE")
    print("="*60)

    print("\n❌ IF CREDENTIALS FAIL:")
    print("   1. Check your .env file has correct values")
    print("   2. Verify API permissions in Bitget")
    print("   3. Make sure API key is not expired")
    print("   4. Check your internet connection")

    print("\n❌ IF BALANCE IS LOW:")
    print("   1. Deposit USDT to your Bitget account")
    print("   2. Wait for deposit confirmation")
    print("   3. Refresh the dashboard")

    print("\n❌ IF MARKET DATA FAILS:")
    print("   1. Check Bitget API status")
    print("   2. Verify API key has read permissions")
    print("   3. Check internet connection")

    print("\n❌ IF AUTOTRADING FAILS:")
    print("   1. Ensure balance >= 5 USDT")
    print("   2. Check API credentials are working")
    print("   3. Verify trading permissions enabled")
    print("   4. Check system logs for errors")

    print("\n📞 SUPPORT:")
    print("   • Check Bitget API documentation")
    print("   • Verify your account permissions")
    print("   • Test with small amounts first")

def main():
    """Main test function"""
    print("🚀 ALE AI Trading - API Connection Test")
    print("=" * 45)

    # Run all tests
    tests = [
        ("Environment File", test_env_file),
        ("Bitget Connection", test_bitget_connection),
        ("Market Data", test_market_data),
        ("Autotrading Readiness", test_autotrading_readiness)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{status}")
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            results.append(False)

    # Summary
    passed = sum(results)
    total = len(results)

    print("\n" + "="*60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your autotrading system is READY!")
        print("\n🚀 NEXT STEPS:")
        print("   1. python web_ui_singularity.py")
        print("   2. Open http://localhost:5000")
        print("   3. Enable autotrading from dashboard")
        print("   4. Monitor your automated trades!")
    else:
        print(f"⚠️ {total - passed} tests failed")
        print("❌ Please fix the issues above before trading")
        provide_troubleshooting()

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
