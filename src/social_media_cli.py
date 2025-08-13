"""
Social Media CLI for cross-platform video management.
Provides command-line interface for TikTok, Instagram, and YouTube operations.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from src.config.settings import get_config
from src.integrations.social_media_manager import (
    SocialMediaManager, 
    create_social_media_manager,
    CrossPlatformVideoMetadata,
    PlatformType
)


class SocialMediaCLI:
    """Command-line interface for social media operations."""
    
    def __init__(self):
        self.config = get_config()
        self.manager = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize the social media manager."""
        try:
            self.manager = create_social_media_manager(self.config)
            self.logger.info("Social media manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize social media manager: {e}")
            return False
        return True
    
    async def upload_video(self, video_path: str, title: str, description: str = None, 
                          platforms: List[str] = None, tags: List[str] = None,
                          scheduled_time: str = None, cross_post_delay: int = 300):
        """Upload video to multiple platforms."""
        if not self.manager:
            print("❌ Social media manager not initialized")
            return False
        
        # Validate video file
        video_file = Path(video_path)
        if not video_file.exists():
            print(f"❌ Video file not found: {video_path}")
            return False
        
        # Parse platforms
        if not platforms:
            platforms = [PlatformType.YOUTUBE]
        else:
            platforms = [PlatformType(p.lower()) for p in platforms]
        
        # Parse scheduled time
        scheduled_dt = None
        if scheduled_time:
            try:
                scheduled_dt = datetime.fromisoformat(scheduled_time)
            except ValueError:
                print(f"❌ Invalid scheduled time format: {scheduled_time}")
                return False
        
        # Create metadata
        metadata = CrossPlatformVideoMetadata(
            title=title,
            description=description,
            tags=tags or [],
            platforms=platforms,
            scheduled_time=scheduled_dt,
            cross_post_delay=cross_post_delay
        )
        
        print(f"🚀 Starting upload to platforms: {', '.join(platforms)}")
        print(f"📹 Video: {video_path}")
        print(f"📝 Title: {title}")
        if description:
            print(f"📄 Description: {description}")
        if tags:
            print(f"🏷️ Tags: {', '.join(tags)}")
        if scheduled_dt:
            print(f"⏰ Scheduled for: {scheduled_dt}")
        print()
        
        try:
            results = await self.manager.upload_to_platforms(video_file, metadata)
            
            # Display results
            print("📊 Upload Results:")
            print("=" * 50)
            
            for result in results:
                platform_icon = self._get_platform_icon(result.platform)
                status_icon = "✅" if result.success else "❌"
                
                print(f"{platform_icon} {result.platform.value.upper()}: {status_icon}")
                
                if result.success:
                    print(f"   🆔 Media ID: {result.media_id}")
                    print(f"   🔗 Share URL: {result.share_url}")
                    print(f"   ⏰ Upload Time: {result.upload_time}")
                else:
                    print(f"   ❌ Error: {result.error_message}")
                print()
            
            return True
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return False
    
    async def check_status(self, platform: str = None):
        """Check upload status and statistics."""
        if not self.manager:
            print("❌ Social media manager not initialized")
            return False
        
        try:
            # Get statistics
            stats = await self.manager.get_upload_statistics()
            
            print("📊 Upload Statistics:")
            print("=" * 50)
            print(f"📈 Total Uploads: {stats['total_uploads']}")
            print(f"✅ Successful: {stats['successful_uploads']}")
            print(f"❌ Failed: {stats['failed_uploads']}")
            print(f"📊 Success Rate: {stats['success_rate']:.1f}%")
            print()
            
            # Platform-specific stats
            if stats['platform_stats']:
                print("📱 Platform Statistics:")
                print("-" * 30)
                for platform, platform_stats in stats['platform_stats'].items():
                    platform_icon = self._get_platform_icon(platform)
                    success_rate = platform_stats['success_rate']
                    print(f"{platform_icon} {platform.value.upper()}: {success_rate:.1f}% ({platform_stats['successful']}/{platform_stats['total']})")
                print()
            
            # Recent uploads
            history = await self.manager.get_upload_history(limit=10)
            if history:
                print("🕒 Recent Uploads:")
                print("-" * 30)
                for entry in history:
                    platform_icon = self._get_platform_icon(entry['platform'])
                    status_icon = "✅" if entry['result'].success else "❌"
                    timestamp = entry['timestamp'].strftime("%Y-%m-%d %H:%M")
                    
                    print(f"{platform_icon} {entry['platform'].value.upper()} - {timestamp} {status_icon}")
                    if entry['result'].success:
                        print(f"   📹 {entry['video_path']}")
                        if entry['result'].share_url:
                            print(f"   🔗 {entry['result'].share_url}")
                    else:
                        print(f"   ❌ {entry['result'].error_message}")
                    print()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to get status: {e}")
            return False
    
    async def get_platform_info(self, platform: str):
        """Get information about a specific platform."""
        if not self.manager:
            print("❌ Social media manager not initialized")
            return False
        
        try:
            platform_enum = PlatformType(platform.lower())
            
            # Get platform limits
            limits = self.manager.get_platform_limits(platform_enum)
            if limits:
                print(f"📱 {platform.upper()} Platform Limits:")
                print("=" * 40)
                print(f"📁 Max File Size: {limits['max_file_size_mb']} MB")
                print(f"⏱️ Max Duration: {limits['max_duration_seconds']} seconds")
                print(f"📅 Daily Upload Limit: {limits['daily_upload_limit']}")
                print()
            
            # Get optimal timing
            timing = self.manager.get_optimal_posting_times(platform_enum)
            if timing:
                print(f"⏰ Optimal Posting Times for {platform.upper()}:")
                print("=" * 50)
                print(f"📅 Best Days: {', '.join(timing['best_days'])}")
                print(f"🕐 Best Hours: {', '.join(map(str, timing['best_hours']))}")
                print(f"🌍 Timezone: {timing['timezone']}")
                print()
            
            return True
            
        except ValueError:
            print(f"❌ Invalid platform: {platform}")
            print(f"Available platforms: {', '.join([p.value for p in PlatformType])}")
            return False
        except Exception as e:
            print(f"❌ Failed to get platform info: {e}")
            return False
    
    async def cleanup_history(self, days: int = 30):
        """Clean up old upload history."""
        if not self.manager:
            print("❌ Social media manager not initialized")
            return False
        
        try:
            await self.manager.cleanup_old_uploads(days)
            print(f"🧹 Cleaned up upload history older than {days} days")
            return True
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
            return False
    
    def _get_platform_icon(self, platform: PlatformType) -> str:
        """Get platform-specific icon."""
        icons = {
            PlatformType.YOUTUBE: "📺",
            PlatformType.TIKTOK: "🎵",
            PlatformType.INSTAGRAM: "📷"
        }
        return icons.get(platform, "📱")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Social Media CLI for cross-platform video management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload video to all platforms
  python social_media_cli.py upload video.mp4 "My Amazing Video" --platforms youtube tiktok instagram
  
  # Upload with custom metadata
  python social_media_cli.py upload video.mp4 "My Video" --description "Check this out!" --tags funny viral
  
  # Schedule upload for later
  python social_media_cli.py upload video.mp4 "My Video" --scheduled "2024-01-15T15:00:00"
  
  # Check upload status
  python social_media_cli.py status
  
  # Get platform information
  python social_media_cli.py platform tiktok
  
  # Clean up old history
  python social_media_cli.py cleanup --days 30
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload video to social media platforms')
    upload_parser.add_argument('video_path', help='Path to video file')
    upload_parser.add_argument('title', help='Video title')
    upload_parser.add_argument('--description', help='Video description')
    upload_parser.add_argument('--platforms', nargs='+', choices=['youtube', 'tiktok', 'instagram'],
                              help='Target platforms (default: youtube)')
    upload_parser.add_argument('--tags', nargs='+', help='Hashtags for the video')
    upload_parser.add_argument('--scheduled', help='Scheduled upload time (ISO format)')
    upload_parser.add_argument('--cross-post-delay', type=int, default=300,
                              help='Delay between platform posts in seconds (default: 300)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check upload status and statistics')
    
    # Platform info command
    platform_parser = subparsers.add_parser('platform', help='Get platform information')
    platform_parser.add_argument('platform', choices=['youtube', 'tiktok', 'instagram'],
                                help='Platform to get info for')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old upload history')
    cleanup_parser.add_argument('--days', type=int, default=30,
                               help='Days of history to keep (default: 30)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = SocialMediaCLI()
    if not await cli.initialize():
        print("❌ Failed to initialize social media manager")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'upload':
            success = await cli.upload_video(
                video_path=args.video_path,
                title=args.title,
                description=args.description,
                platforms=args.platforms,
                tags=args.tags,
                scheduled_time=args.scheduled,
                cross_post_delay=args.cross_post_delay
            )
        elif args.command == 'status':
            success = await cli.check_status()
        elif args.command == 'platform':
            success = await cli.get_platform_info(args.platform)
        elif args.command == 'cleanup':
            success = await cli.cleanup_history(args.days)
        else:
            print(f"❌ Unknown command: {args.command}")
            success = False
        
        if success:
            print("✅ Operation completed successfully")
        else:
            print("❌ Operation failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Operation interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())