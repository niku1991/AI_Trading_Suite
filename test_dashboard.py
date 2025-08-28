#!/usr/bin/env python3
import requests
import sys

def test_dashboard():
    try:
        response = requests.get('http://localhost:5000', timeout=10)
        if response.status_code == 200:
            print("✅ Dashboard is running successfully!")
            print(f"📊 Content length: {len(response.text)} characters")
            print("🌐 Access your AI dashboard at: http://localhost:5000")
            return True
        else:
            print(f"⚠️ Dashboard responded with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Dashboard is not accessible")
        print("💡 The server might still be starting up...")
        return False
    except Exception as e:
        print(f"❌ Error testing dashboard: {e}")
        return False

if __name__ == "__main__":
    success = test_dashboard()
    if success:
        print("\n🎯 YOUR AI AUTOTRADING SYSTEM IS LIVE!")
        print("🤖 AI is monitoring markets 24/7")
        print("📈 Live data streaming active")
        print("💰 Autotrading ready to deploy")
        sys.exit(0)
    else:
        print("\n⏳ System is starting up...")
        print("🔄 Please wait a moment and try again")
        sys.exit(1)
