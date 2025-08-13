"""
Unified Social Media Manager for cross-platform video distribution.
Coordinates uploads across YouTube, TikTok, and Instagram with intelligent scheduling.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel, Field

from .youtube_client import YouTubeClient
from .tiktok_client import TikTokClient, TikTokVideoMetadata
from .instagram_client import InstagramClient, InstagramVideoMetadata


class PlatformType(str, Enum):
    """Supported social media platforms."""
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"


class UploadStatus(str, Enum):
    """Upload status enumeration."""
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SCHEDULED = "scheduled"


@dataclass
class CrossPlatformUploadResult:
    """Result of a cross-platform upload operation."""
    platform: PlatformType
    success: bool
    media_id: Optional[str] = None
    share_url: Optional[str] = None
    error_message: Optional[str] = None
    upload_time: Optional[datetime] = None
    processing_status: UploadStatus = UploadStatus.PENDING
    metadata: Optional[Dict[str, Any]] = None


class CrossPlatformVideoMetadata(BaseModel):
    """Unified metadata for cross-platform video uploads."""
    title: str = Field(..., description="Video title")
    description: Optional[str] = Field(None, description="Video description")
    tags: List[str] = Field(default_factory=list, description="Hashtags")
    
    # Platform-specific customizations
    youtube_metadata: Optional[Dict[str, Any]] = Field(None, description="YouTube-specific settings")
    tiktok_metadata: Optional[Dict[str, Any]] = Field(None, description="TikTok-specific settings")
    instagram_metadata: Optional[Dict[str, Any]] = Field(None, description="Instagram-specific settings")
    
    # Cross-platform settings
    scheduled_time: Optional[datetime] = Field(None, description="Scheduled upload time")
    platforms: List[PlatformType] = Field(default_factory=lambda: [PlatformType.YOUTUBE])
    cross_post_delay: int = Field(300, description="Delay between platform posts in seconds")


class SocialMediaManager:
    """
    Unified manager for cross-platform social media uploads.
    
    Handles:
    - Multi-platform video distribution
    - Intelligent scheduling and timing
    - Platform-specific optimization
    - Upload monitoring and retry logic
    - Cross-platform analytics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize platform clients
        self.clients = {}
        self._initialize_clients()
        
        # Upload queue and status tracking
        self.upload_queue = asyncio.Queue()
        self.upload_status = {}
        self.upload_history = []
        
        # Scheduling and optimization
        self.optimal_timing = self._load_optimal_timing()
        self.platform_limits = self._load_platform_limits()
        
        # Performance tracking
        self.upload_stats = {
            'total_uploads': 0,
            'successful_uploads': 0,
            'failed_uploads': 0,
            'platform_stats': {}
        }
        
        self.logger.info("Social Media Manager initialized")
    
    def _initialize_clients(self):
        """Initialize platform-specific clients."""
        try:
            # YouTube client
            if 'youtube_api_key' in self.config:
                from .youtube_client import create_youtube_client
                self.clients[PlatformType.YOUTUBE] = create_youtube_client(self.config)
                self.logger.info("YouTube client initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize YouTube client: {e}")
        
        try:
            # TikTok client
            if 'tiktok_api_key' in self.config:
                from .tiktok_client import create_tiktok_client
                self.clients[PlatformType.TIKTOK] = create_tiktok_client(self.config)
                self.logger.info("TikTok client initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize TikTok client: {e}")
        
        try:
            # Instagram client
            if 'instagram_username' in self.config:
                from .instagram_client import create_instagram_client
                self.clients[PlatformType.INSTAGRAM] = create_instagram_client(self.config)
                self.logger.info("Instagram client initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Instagram client: {e}")
        
        if not self.clients:
            self.logger.error("No social media clients available")
    
    def _load_optimal_timing(self) -> Dict[str, Any]:
        """Load optimal posting times for each platform."""
        return {
            PlatformType.YOUTUBE: {
                'best_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                'best_hours': [12, 15, 19, 21],  # 12 PM, 3 PM, 7 PM, 9 PM
                'timezone': 'UTC'
            },
            PlatformType.TIKTOK: {
                'best_days': ['Tuesday', 'Thursday', 'Friday', 'Saturday'],
                'best_hours': [9, 12, 19, 21],  # 9 AM, 12 PM, 7 PM, 9 PM
                'timezone': 'UTC'
            },
            PlatformType.INSTAGRAM: {
                'best_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                'best_hours': [11, 13, 15, 19],  # 11 AM, 1 PM, 3 PM, 7 PM
                'timezone': 'UTC'
            }
        }
    
    def _load_platform_limits(self) -> Dict[str, Any]:
        """Load platform-specific upload limits."""
        return {
            PlatformType.YOUTUBE: {
                'max_file_size_mb': 128,
                'max_duration_seconds': 43200,  # 12 hours
                'daily_upload_limit': 100
            },
            PlatformType.TIKTOK: {
                'max_file_size_mb': 287,
                'max_duration_seconds': 600,  # 10 minutes
                'daily_upload_limit': 50
            },
            PlatformType.INSTAGRAM: {
                'max_file_size_mb': 100,
                'max_duration_seconds': 90,  # 1.5 minutes for Reels
                'daily_upload_limit': 25
            }
        }
    
    async def upload_to_platforms(self, video_path: Union[str, Path], 
                                 metadata: CrossPlatformVideoMetadata) -> List[CrossPlatformUploadResult]:
        """
        Upload video to multiple platforms with intelligent scheduling.
        
        Args:
            video_path: Path to the video file
            metadata: Cross-platform metadata and settings
        
        Returns:
            List of upload results for each platform
        """
        results = []
        video_path = Path(video_path)
        
        # Validate video file
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check platform availability
        available_platforms = [p for p in metadata.platforms if p in self.clients]
        if not available_platforms:
            raise ValueError("No available platforms for upload")
        
        # Determine optimal upload timing
        optimal_times = self._calculate_optimal_times(available_platforms, metadata.scheduled_time)
        
        # Upload to each platform
        for i, platform in enumerate(available_platforms):
            try:
                # Calculate delay for cross-posting
                if i > 0 and metadata.cross_post_delay > 0:
                    delay = metadata.cross_post_delay * i
                    await asyncio.sleep(delay)
                
                # Get platform-specific metadata
                platform_metadata = self._get_platform_metadata(platform, metadata)
                
                # Upload to platform
                result = await self._upload_to_platform(platform, video_path, platform_metadata)
                results.append(result)
                
                # Update statistics
                self._update_upload_stats(platform, result.success)
                
                # Store in history
                self.upload_history.append({
                    'platform': platform,
                    'video_path': str(video_path),
                    'result': result,
                    'timestamp': datetime.now()
                })
                
            except Exception as e:
                self.logger.error(f"Upload to {platform} failed: {e}")
                results.append(CrossPlatformUploadResult(
                    platform=platform,
                    success=False,
                    error_message=str(e),
                    processing_status=UploadStatus.FAILED
                ))
        
        return results
    
    def _calculate_optimal_times(self, platforms: List[PlatformType], 
                                scheduled_time: Optional[datetime]) -> Dict[PlatformType, datetime]:
        """Calculate optimal upload times for each platform."""
        if scheduled_time:
            # Use provided scheduled time
            return {platform: scheduled_time for platform in platforms}
        
        # Calculate optimal times based on platform-specific best practices
        optimal_times = {}
        base_time = datetime.now()
        
        for platform in platforms:
            platform_timing = self.optimal_timing.get(platform, {})
            best_hours = platform_timing.get('best_hours', [12])
            
            # Find the next best hour
            current_hour = base_time.hour
            next_best_hour = None
            
            for hour in best_hours:
                if hour > current_hour:
                    next_best_hour = hour
                    break
            
            if next_best_hour is None:
                # Use first best hour tomorrow
                next_best_hour = best_hours[0]
                base_time += timedelta(days=1)
            
            # Set to next best hour
            optimal_time = base_time.replace(
                hour=next_best_hour,
                minute=0,
                second=0,
                microsecond=0
            )
            
            optimal_times[platform] = optimal_time
        
        return optimal_times
    
    def _get_platform_metadata(self, platform: PlatformType, 
                               metadata: CrossPlatformVideoMetadata) -> Dict[str, Any]:
        """Get platform-specific metadata."""
        base_metadata = {
            'title': metadata.title,
            'description': metadata.description,
            'tags': metadata.tags
        }
        
        if platform == PlatformType.YOUTUBE:
            platform_specific = metadata.youtube_metadata or {}
            return {**base_metadata, **platform_specific}
        
        elif platform == PlatformType.TIKTOK:
            platform_specific = metadata.tiktok_metadata or {}
            return {**base_metadata, **platform_specific}
        
        elif platform == PlatformType.INSTAGRAM:
            platform_specific = metadata.instagram_metadata or {}
            return {**base_metadata, **platform_specific}
        
        return base_metadata
    
    async def _upload_to_platform(self, platform: PlatformType, video_path: Path, 
                                 metadata: Dict[str, Any]) -> CrossPlatformUploadResult:
        """Upload video to a specific platform."""
        client = self.clients.get(platform)
        if not client:
            raise ValueError(f"No client available for platform: {platform}")
        
        try:
            if platform == PlatformType.YOUTUBE:
                # YouTube upload
                result = await client.upload_video(str(video_path), metadata)
                return CrossPlatformUploadResult(
                    platform=platform,
                    success=result.get('success', False),
                    media_id=result.get('video_id'),
                    share_url=result.get('share_url'),
                    error_message=result.get('error_message'),
                    upload_time=datetime.now(),
                    processing_status=UploadStatus.COMPLETED if result.get('success') else UploadStatus.FAILED,
                    metadata=metadata
                )
            
            elif platform == PlatformType.TIKTOK:
                # TikTok upload
                tiktok_metadata = TikTokVideoMetadata(**metadata)
                result = await client.upload_video(video_path, tiktok_metadata)
                return CrossPlatformUploadResult(
                    platform=platform,
                    success=result.success,
                    media_id=result.video_id,
                    share_url=result.share_url,
                    error_message=result.error_message,
                    upload_time=result.upload_time,
                    processing_status=UploadStatus.COMPLETED if result.success else UploadStatus.FAILED,
                    metadata=metadata
                )
            
            elif platform == PlatformType.INSTAGRAM:
                # Instagram upload
                instagram_metadata = InstagramVideoMetadata(**metadata)
                result = await client.upload_video(video_path, instagram_metadata)
                return CrossPlatformUploadResult(
                    platform=platform,
                    success=result.success,
                    media_id=result.media_id,
                    share_url=result.share_url,
                    error_message=result.error_message,
                    upload_time=result.upload_time,
                    processing_status=UploadStatus.COMPLETED if result.success else UploadStatus.FAILED,
                    metadata=metadata
                )
            
        except Exception as e:
            self.logger.error(f"Upload to {platform} failed: {e}")
            return CrossPlatformUploadResult(
                platform=platform,
                success=False,
                error_message=str(e),
                processing_status=UploadStatus.FAILED,
                metadata=metadata
            )
    
    def _update_upload_stats(self, platform: PlatformType, success: bool):
        """Update upload statistics."""
        self.upload_stats['total_uploads'] += 1
        
        if success:
            self.upload_stats['successful_uploads'] += 1
        else:
            self.upload_stats['failed_uploads'] += 1
        
        if platform not in self.upload_stats['platform_stats']:
            self.upload_stats['platform_stats'][platform] = {
                'total': 0,
                'successful': 0,
                'failed': 0
            }
        
        self.upload_stats['platform_stats'][platform]['total'] += 1
        if success:
            self.upload_stats['platform_stats'][platform]['successful'] += 1
        else:
            self.upload_stats['platform_stats'][platform]['failed'] += 1
    
    async def get_upload_status(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific upload."""
        return self.upload_status.get(upload_id)
    
    async def get_upload_history(self, platform: Optional[PlatformType] = None, 
                                limit: int = 50) -> List[Dict[str, Any]]:
        """Get upload history, optionally filtered by platform."""
        history = self.upload_history
        
        if platform:
            history = [h for h in history if h['platform'] == platform]
        
        return sorted(history, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    async def get_upload_statistics(self) -> Dict[str, Any]:
        """Get comprehensive upload statistics."""
        return {
            **self.upload_stats,
            'success_rate': (
                self.upload_stats['successful_uploads'] / max(self.upload_stats['total_uploads'], 1)
            ) * 100,
            'platform_success_rates': {
                platform: {
                    'total': stats['total'],
                    'success_rate': (stats['successful'] / max(stats['total'], 1)) * 100
                }
                for platform, stats in self.upload_stats['platform_stats'].items()
            }
        }
    
    async def retry_failed_upload(self, upload_id: str) -> Optional[CrossPlatformUploadResult]:
        """Retry a failed upload."""
        # Implementation for retry logic
        pass
    
    async def schedule_upload(self, video_path: Union[str, Path], 
                             metadata: CrossPlatformVideoMetadata, 
                             scheduled_time: datetime) -> str:
        """Schedule an upload for a future time."""
        # Implementation for scheduled uploads
        pass
    
    def get_platform_limits(self, platform: PlatformType) -> Dict[str, Any]:
        """Get upload limits for a specific platform."""
        return self.platform_limits.get(platform, {})
    
    def get_optimal_posting_times(self, platform: PlatformType) -> Dict[str, Any]:
        """Get optimal posting times for a specific platform."""
        return self.optimal_timing.get(platform, {})
    
    async def cleanup_old_uploads(self, days_to_keep: int = 30):
        """Clean up old upload history entries."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        self.upload_history = [
            h for h in self.upload_history 
            if h['timestamp'] > cutoff_date
        ]
        self.logger.info(f"Cleaned up upload history older than {days_to_keep} days")


def create_social_media_manager(config: Dict[str, Any]) -> SocialMediaManager:
    """Factory function to create a social media manager."""
    return SocialMediaManager(config)