#!/usr/bin/env python3
"""
Test script for the new TBot AI-powered command line tool
"""

import sys
import tempfile
from pathlib import Path

# Add tbot to path
sys.path.insert(0, str(Path(__file__).parent))

from tbot.main import TBotCLI


def test_tbot_cli():
    """Test the TBot CLI functionality"""
    print("🧪 Testing TBot AI-Powered CLI Tool")
    print("=" * 50)
    
    # Create TBot CLI instance
    tbot = TBotCLI()
    
    # Test 1: Health check
    print("\n1. Health Check Test")
    healthy = tbot.health_check()
    print(f"   Health check result: {'✅ Healthy' if healthy else '❌ Unhealthy'}")
    
    if not healthy:
        print("⚠️  Servers not available, skipping functionality tests")
        return
    
    # Test 2: Summarize functionality
    print("\n2. Summarize Test")
    try:
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
            This is a test document for summarization. It contains multiple paragraphs
            with various information. The first paragraph introduces the topic.
            
            The second paragraph provides more details about the subject matter.
            It explains various concepts and ideas that are relevant to the discussion.
            
            The final paragraph concludes the document with a summary of key points
            and provides some recommendations for future work.
            """)
            temp_file = f.name
        
        print(f"   Testing summarization of {temp_file}")
        tbot.summarize(temp_file, "short")
        
        # Clean up
        Path(temp_file).unlink()
        
    except Exception as e:
        print(f"   ❌ Summarize test failed: {e}")
    
    # Test 3: Text analysis
    print("\n3. Analysis Test")
    try:
        test_text = "This is a sample text for analysis. It contains positive sentiment and technical language."
        print("   Testing sentiment analysis...")
        tbot.analyze(test_text, "sentiment")
    except Exception as e:
        print(f"   ❌ Analysis test failed: {e}")
    
    # Test 4: Text transformation
    print("\n4. Transform Test")
    try:
        test_data = "Name: John, Age: 30, City: New York\nName: Jane, Age: 25, City: Boston"
        print("   Testing conversion to table...")
        tbot.convert_to_table(test_data)
    except Exception as e:
        print(f"   ❌ Transform test failed: {e}")
    
    print("\n🎉 Test completed!")


def main():
    """Main test function"""
    try:
        test_tbot_cli()
    except KeyboardInterrupt:
        print("\n👋 Test interrupted!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    main()
