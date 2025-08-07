"""
Enhanced AI-Powered YouTube Shorts Generator
Main entry point for creating YouTube Shorts from Reddit content with Spotify music integration.
"""

import asyncio
import logging
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.config.settings import get_config, setup_logging
from src.application import Application
from src.utils.app_utils import setup_application, clear_temp_files, clear_results, clear_logs


async def main():
    """Main entry point with CLI interface"""
    parser = argparse.ArgumentParser(
        description="Enhanced AI-Powered YouTube Shorts Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run fully autonomous mode (DEFAULT - NO HUMAN INPUT REQUIRED)
  python main.py
  python main.py autonomous
  python main.py autonomous --max-videos-per-day 10 --min-videos-per-day 5
  
  # Check system status
  python main.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Autonomous mode (NEW DEFAULT)
    autonomous_parser = subparsers.add_parser('autonomous', help='Run fully autonomous video generation (NO HUMAN INPUT)')
    autonomous_parser.add_argument('--stats-interval', type=int, default=3600,
                                  help='Statistics reporting interval in seconds (default: 3600)')
    autonomous_parser.add_argument('--max-videos-per-day', type=int, default=8,
                                  help='Maximum videos to generate per day (default: 8)')
    autonomous_parser.add_argument('--min-videos-per-day', type=int, default=3,
                                  help='Minimum videos to generate per day (default: 3)')
    autonomous_parser.add_argument('--video-check-interval', type=int, default=3600,
                                  help='Interval between video generation checks in seconds (default: 3600)')
    
    # System status
    status_parser = subparsers.add_parser('status', help='Check system status')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up temporary files, results, and logs')
    cleanup_parser.add_argument('--logs', action='store_true', help='Also clear log files')
    cleanup_parser.add_argument('--all', action='store_true', help='Clear all data (temp, results, logs)')
    
    args = parser.parse_args()
    
    if not args.command:
        # Default to 'autonomous' command when no command is specified
        print("🚀 No command specified, starting AUTONOMOUS MODE...")
        print("🤖 System will run continuously with NO HUMAN INPUT required")
        print("📊 Intelligent scheduling and automatic video generation enabled")
        print("⏹️ Press Ctrl+C to stop autonomous mode\n")
        
        # Create a mock args object with default autonomous parameters
        class MockArgs:
            def __init__(self):
                self.command = 'autonomous'
                self.stats_interval = 3600
                self.max_videos_per_day = 8
                self.min_videos_per_day = 3
                self.video_check_interval = 3600
        
        args = MockArgs()
        
        # Log the default autonomous configuration
        print(f"📊 Using default autonomous configuration:")
        print(f"   Max videos per day: {args.max_videos_per_day}")
        print(f"   Min videos per day: {args.min_videos_per_day}")  
        print(f"   Video check interval: {args.video_check_interval}s")
        print(f"   Stats interval: {args.stats_interval}s")
        print("   Use 'python main.py autonomous --help' to see available options\n")
    
    # Initialize application
    setup_application()
    
    try:
        if args.command == 'autonomous':
            # Run fully autonomous mode using new Application class
            print("🚀 Starting Enhanced Autonomous Video Generation System")
            print("🤖 No human input required - system will run continuously")
            print("📊 Intelligent scheduling and optimization enabled")
            print("⏹️ Press Ctrl+C to stop\n")
            
            # Initialize new Application class with autonomous arguments
            autonomous_args = {
                'max_videos_per_day': args.max_videos_per_day,
                'min_videos_per_day': args.min_videos_per_day,
                'video_check_interval': args.video_check_interval,
                'stats_interval': args.stats_interval
            }
            
            app = Application(autonomous_args=autonomous_args)
            
            # Start autonomous mode
            await app.run_autonomous_mode()
            
        elif args.command == 'status':
            # Check system status using Application class
            app = Application()
            await app.get_system_status()
        
        elif args.command == 'cleanup':
            print("🧹 Starting cleanup process...")
            clear_temp_files()
            clear_results()
            if args.logs or args.all:
                clear_logs()
            if args.all:
                # Add any other all-encompassing cleanup here
                pass
            print("✅ Cleanup process finished.")
    
    except KeyboardInterrupt:
        print("\n⏹️ Operation interrupted by user")
        print("Cleaning up resources...")
    except Exception as e:
        print(f"🚨 ERROR: Operation failed: {e}")
        logging.getLogger(__name__).exception("Main operation failed")
    finally:
        # Ensure cleanup always runs
        try:
            pass  # Application handles its own cleanup
        except:
            pass  # Ignore cleanup errors


def run_main():
    """Safe main entry point with proper async handling"""
    try:
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            print("⚠️ Already in async context. Use 'await main()' instead.")
            return 1
        except RuntimeError:
            # No running loop - this is what we want
            pass
        
        # Run the main function
        return asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Startup interrupted by user")
        return 0
    except Exception as e:
        print(f"🚨 Critical startup error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(run_main())