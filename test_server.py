#!/usr/bin/env python3
import requests
import sys

def test_server():
    try:
        response = requests.get('http://localhost:5000', timeout=10)
        print(f"✅ Server is running!")
        print(f"Status Code: {response.status_code}")
        print(f"Content Length: {len(response.text)} characters")
        if response.status_code == 200:
            print("✅ Dashboard is accessible!")
            return True
        else:
            print(f"⚠️  Server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running or not accessible")
        return False
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False

if __name__ == "__main__":
    success = test_server()
    sys.exit(0 if success else 1)
