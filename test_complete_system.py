#!/usr/bin/env python3
"""
Complete System Test for ALE AI Trading Suite
Tests all components including real-time data, balance, and charts
"""

import requests
import json
import time
from datetime import datetime

def test_dashboard():
    """Test dashboard accessibility"""
    print("🌐 Testing Dashboard...")
    try:
        response = requests.get('http://localhost:5000', timeout=10)
        if response.status_code == 200:
            print("✅ Dashboard: Accessible")
            return True
        else:
            print(f"❌ Dashboard: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Dashboard: {e}")
        return False

def test_api_endpoints():
    """Test all API endpoints"""
    print("\n🔗 Testing API Endpoints...")
    
    endpoints = [
        ('/api/system/status', 'System Status'),
        ('/api/trading/status', 'Trading Status'),
        ('/api/market/analysis', 'Market Analysis'),
        ('/api/trading/portfolio', 'Portfolio Data'),
        ('/api/ai/status', 'AI Status'),
        ('/api/bitget/test-connection', 'Bitget Connection'),
        ('/api/bitget/balance', 'Bitget Balance'),
        ('/api/market/ticker/BTCUSDT', 'BTC Ticker'),
        ('/api/market/ticker/ETHUSDT', 'ETH Ticker'),
        ('/api/persistence/status', 'Persistence Status')
    ]
    
    results = {}
    for endpoint, name in endpoints:
        try:
            response = requests.get(f'http://localhost:5000{endpoint}', timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {name}: Working")
                results[endpoint] = {'status': 'success', 'data': data}
            else:
                print(f"❌ {name}: HTTP {response.status_code}")
                results[endpoint] = {'status': 'error', 'code': response.status_code}
        except Exception as e:
            print(f"❌ {name}: {e}")
            results[endpoint] = {'status': 'error', 'message': str(e)}
    
    return results

def test_real_time_data():
    """Test real-time market data"""
    print("\n📊 Testing Real-Time Data...")
    
    try:
        # Test BTC ticker
        response = requests.get('http://localhost:5000/api/market/ticker/BTCUSDT', timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                ticker_data = data.get('data', {})
                price = ticker_data.get('last', 'N/A')
                volume = ticker_data.get('baseVolume', 'N/A')
                print(f"✅ BTC/USDT: ${price} (Volume: {volume})")
                return True
            else:
                print(f"❌ BTC Ticker: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ BTC Ticker: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Real-time data test failed: {e}")
        return False

def test_bitget_balance():
    """Test Bitget account balance"""
    print("\n💰 Testing Bitget Balance...")
    
    try:
        response = requests.get('http://localhost:5000/api/bitget/balance', timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                balance_data = data.get('data', [])
                print(f"✅ Balance API: Working")
                print(f"📊 Account Assets: {len(balance_data)} assets")
                
                # Find USDT balance
                usdt_balance = 0
                for asset in balance_data:
                    if asset.get('coin') == 'USDT':
                        usdt_balance = float(asset.get('available', 0))
                        break
                
                print(f"💰 USDT Balance: ${usdt_balance:.2f}")
                return True
            else:
                print(f"❌ Balance API: {data.get('message', 'Unknown error')}")
                return False
        else:
            print(f"❌ Balance API: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Balance test failed: {e}")
        return False

def test_ai_systems():
    """Test AI systems status"""
    print("\n🧠 Testing AI Systems...")
    
    try:
        response = requests.get('http://localhost:5000/api/ai/status', timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                ai_data = data.get('data', {})
                ai_systems = ai_data.get('ai_systems', {})
                
                print("✅ AI Systems Status:")
                for system, status in ai_systems.items():
                    active = status.get('active', False)
                    status_icon = "🟢" if active else "🔴"
                    print(f"   {status_icon} {system.title()}: {'Active' if active else 'Inactive'}")
                
                consciousness = ai_data.get('consciousness', {})
                level = consciousness.get('current_level', 0)
                print(f"🧠 Consciousness Level: {level:.1%}")
                
                return True
            else:
                print(f"❌ AI Status: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ AI Status: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ AI systems test failed: {e}")
        return False

def test_autotrading():
    """Test autotrading functionality"""
    print("\n🤖 Testing Autotrading...")
    
    try:
        # Test autotrading status
        response = requests.get('http://localhost:5000/api/trading/status', timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                trading_data = data.get('data', {})
                active = trading_data.get('trading_active', False)
                positions = trading_data.get('positions', 0)
                orders = trading_data.get('active_orders', 0)
                
                print(f"✅ Trading Status: {'Active' if active else 'Inactive'}")
                print(f"📊 Active Positions: {positions}")
                print(f"📋 Active Orders: {orders}")
                
                return True
            else:
                print(f"❌ Trading Status: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Trading Status: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Autotrading test failed: {e}")
        return False

def test_persistence():
    """Test data persistence"""
    print("\n💾 Testing Data Persistence...")
    
    try:
        response = requests.get('http://localhost:5000/api/persistence/status', timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                persistence_data = data.get('data', {})
                active = persistence_data.get('persistence_active', False)
                interval = persistence_data.get('auto_save_interval', 'Unknown')
                summary = persistence_data.get('summary', {})
                
                print(f"✅ Persistence: {'Active' if active else 'Inactive'}")
                print(f"⏰ Auto-save Interval: {interval}")
                print(f"📊 Database Records: {summary.get('database_records', 0)}")
                print(f"📁 Total Files: {summary.get('total_files', 0)}")
                
                return True
            else:
                print(f"❌ Persistence: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Persistence: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Persistence test failed: {e}")
        return False

def main():
    """Run complete system test"""
    print("🚀 ALE AI Trading Suite - Complete System Test")
    print("=" * 60)
    print(f"🕐 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    tests = [
        ("Dashboard", test_dashboard),
        ("API Endpoints", lambda: test_api_endpoints()),
        ("Real-Time Data", test_real_time_data),
        ("Bitget Balance", test_bitget_balance),
        ("AI Systems", test_ai_systems),
        ("Autotrading", test_autotrading),
        ("Data Persistence", test_persistence)
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"✅ Tests Passed: {passed}/{total}")
    print(f"❌ Tests Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your AI Trading Suite is fully operational!")
        print("🌐 Dashboard: http://localhost:5000")
        print("🤖 Ready for live trading with real Bitget data!")
    else:
        print(f"\n⚠️ {total - passed} tests failed")
        print("🔧 Some components may need attention")
    
    print(f"\n🕐 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
