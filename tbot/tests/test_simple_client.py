#!/usr/bin/env python3
"""
Simple test for the new TBot HTTP client using standard library
"""

import sys
from pathlib import Path

# Add tbot to path
sys.path.insert(0, str(Path(__file__).parent))

from tbot.http_client import TBotClient


def test_client():
    """Test the simple HTTP client"""
    print("🧪 Testing TBot HTTP Client...")
    
    # Create client
    client = TBotClient()
    
    print("✅ Client created successfully")
    
    # Test health check (this will fail if Conductor isn't running, which is expected)
    try:
        health = client.health_check()
        print(f"📊 Health check result: {health}")
    except Exception as e:
        print(f"⚠️  Health check failed (expected if Conductor not running): {e}")
    
    print("✅ Test completed")


if __name__ == "__main__":
    test_client()
