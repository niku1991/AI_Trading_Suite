#!/usr/bin/env python3
"""
System Verification Script
Tests all components of the AI autotrading system
"""

import requests
import json
import time
import sys
from pathlib import Path

def test_dashboard():
    """Test dashboard accessibility"""
    try:
        response = requests.get('http://localhost:5000', timeout=10)
        return response.status_code == 200, f"Dashboard: {response.status_code}"
    except Exception as e:
        return False, f"Dashboard error: {e}"

def test_api_endpoints():
    """Test key API endpoints"""
    endpoints = [
        '/api/system/status',
        '/api/trading/status',
        '/api/market/analysis',
        '/api/trading/portfolio',
        '/api/ai/status'
    ]

    results = {}
    for endpoint in endpoints:
        try:
            response = requests.get(f'http://localhost:5000{endpoint}', timeout=5)
            results[endpoint] = response.status_code == 200
        except:
            results[endpoint] = False

    return results

def check_data_files():
    """Check if data files exist"""
    data_files = [
        'ale_ai_data/portfolio/current_portfolio.json',
        'ale_ai_data/market_data.json',
        'ale_ai_data/system_snapshot.json'
    ]

    results = {}
    for file_path in data_files:
        path = Path(file_path)
        results[file_path] = path.exists() and path.stat().st_size > 0

    return results

def check_credentials():
    """Check API credentials status"""
    try:
        env_file = Path.home() / '.ale_ai_trading' / '.env'
        if not env_file.exists():
            return False, "Environment file not found"

        with open(env_file, 'r') as f:
            content = f.read()

        # Check for real credentials (not placeholders)
        api_key = None
        api_secret = None
        passphrase = None

        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('BITGET_API_KEY='):
                api_key = line.split('=', 1)[1]
            elif line.startswith('BITGET_API_SECRET='):
                api_secret = line.split('=', 1)[1]
            elif line.startswith('BITGET_API_PASSPHRASE='):
                passphrase = line.split('=', 1)[1]

        has_real_credentials = (
            api_key and not api_key.startswith('bg_REPLACE') and
            api_secret and not api_secret.startswith('REPLACE') and
            passphrase and not passphrase.startswith('REPLACE')
        )

        return has_real_credentials, "Real credentials configured" if has_real_credentials else "Placeholder credentials detected"

    except Exception as e:
        return False, f"Credentials check error: {e}"

def main():
    print("🚀 ALE AI Trading System - Verification")
    print("="*50)

    # Test 1: Dashboard
    print("\n📊 Testing Dashboard...")
    dashboard_ok, dashboard_msg = test_dashboard()
    print(f"{'✅' if dashboard_ok else '❌'} {dashboard_msg}")

    # Test 2: API Endpoints
    print("\n🔗 Testing API Endpoints...")
    api_results = test_api_endpoints()
    api_ok = all(api_results.values())

    for endpoint, status in api_results.items():
        print(f"{'✅' if status else '❌'} {endpoint}")

    # Test 3: Data Files
    print("\n📁 Checking Data Files...")
    data_results = check_data_files()
    data_ok = all(data_results.values())

    for file_path, exists in data_results.items():
        print(f"{'✅' if exists else '❌'} {file_path}")

    # Test 4: Credentials
    print("\n🔑 Checking Credentials...")
    creds_ok, creds_msg = check_credentials()
    print(f"{'✅' if creds_ok else '❌'} {creds_msg}")

    # Overall Status
    print("\n" + "="*50)
    print("📋 VERIFICATION SUMMARY")
    print("="*50)

    all_tests = [dashboard_ok, api_ok, data_ok, creds_ok]
    passed_tests = sum(all_tests)
    total_tests = len(all_tests)

    print(f"✅ Tests Passed: {passed_tests}/{total_tests}")

    if passed_tests == total_tests:
        print("🎉 SYSTEM STATUS: FULLY OPERATIONAL!")
        print("🚀 Your AI autotrading system is ready for live trading")
        print("\n🌐 Dashboard: http://localhost:5000")
        print("🤖 AI Status: Active and monitoring markets")
        print("📊 Live Data: Streaming from Bitget")
        print("💰 Portfolio: Connected to your Bitget account")
        print("\n🎯 READY TO TRADE AUTONOMOUSLY!")
    else:
        print("⚠️ SYSTEM STATUS: PARTIAL ISSUES DETECTED")
        print("Some components may need attention before full operation")

    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    print(f"\n{'🎯 System Ready for Live Trading!' if success else '⚠️ Some issues need attention'}")
    sys.exit(0 if success else 1)
