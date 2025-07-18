#!/usr/bin/env python3
"""
Autonomous YouTube Video Generator Starter
Simple script to start the autonomous mode with no human input required
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.autonomous_mode import start_autonomous_mode


def main():
    """Main entry point for autonomous mode"""
    print("🚀 YouTube Video Generator - Autonomous Mode")
    print("=" * 50)
    print("🤖 Starting fully autonomous operation...")
    print("📊 No human input required")
    print("⏹️ Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        # Setup basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Start autonomous mode
        asyncio.run(start_autonomous_mode())
        
    except KeyboardInterrupt:
        print("\n⏹️ Autonomous mode stopped by user")
        print("✅ Shutdown complete")
        sys.exit(0)
    except Exception as e:
        print(f"\n🚨 Critical error: {e}")
        print("❌ Autonomous mode failed to start")
        sys.exit(1)


if __name__ == "__main__":
    main()