"""
Scheduler class responsible for determining when to run video generation tasks
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional


class Scheduler:
    """
    Handles scheduling logic for autonomous video generation.
    Single responsibility: Determine WHEN to run tasks based on configuration.
    """
    
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Extract autonomous configuration
        # Try to get autonomous config from config object or fallback to defaults
        if hasattr(config, 'autonomous') and hasattr(config.autonomous, '__dict__'):
            autonomous_config = config.autonomous.__dict__
        else:
            # Use defaults if no autonomous config is available
            autonomous_config = {}
        
        # Scheduling settings with defaults
        self.optimal_posting_times = autonomous_config.get('optimal_posting_times', [9, 12, 16, 19, 21])
        self.video_generation_interval = autonomous_config.get('video_generation_interval', 3600)
        self.min_videos_per_day = autonomous_config.get('min_videos_per_day', 3)
        self.max_videos_per_day = autonomous_config.get('max_videos_per_day', 8)
        
        # State
        self.daily_video_count = 0
        self.last_reset_date = datetime.now().date()
        
    def should_generate_video(self, current_stats: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determine if a video should be generated now based on scheduling rules.
        
        Args:
            current_stats: Optional current system statistics
            
        Returns:
            bool: True if a video should be generated
        """
        now = datetime.now()
        
        # Reset daily count if it's a new day
        if now.date() > self.last_reset_date:
            self.daily_video_count = 0
            self.last_reset_date = now.date()
            
        # Check if we've reached the daily limit
        if self.daily_video_count >= self.max_videos_per_day:
            self.logger.info(f"Daily video limit reached: {self.daily_video_count}/{self.max_videos_per_day}")
            return False
            
        # Check if we're in an optimal posting time
        current_hour = now.hour
        is_optimal_time = current_hour in self.optimal_posting_times
        
        # If we haven't reached minimum videos for the day, generate even if not optimal time
        if self.daily_video_count < self.min_videos_per_day:
            self.logger.info(f"Under minimum daily videos: {self.daily_video_count}/{self.min_videos_per_day}")
            return True
            
        # Otherwise, only generate during optimal times
        if is_optimal_time:
            self.logger.info(f"Optimal posting time: {current_hour}:00")
            return True
            
        self.logger.debug(f"Not optimal time for posting: {current_hour}:00")
        return False
        
    def get_next_scheduled_time(self) -> datetime:
        """
        Get the next scheduled time for video generation.
        
        Returns:
            datetime: Next scheduled generation time
        """
        now = datetime.now()
        current_hour = now.hour
        
        # Find next optimal posting time today
        for hour in self.optimal_posting_times:
            if hour > current_hour:
                return now.replace(hour=hour, minute=0, second=0, microsecond=0)
                
        # If no more optimal times today, use first optimal time tomorrow
        next_day = now + timedelta(days=1)
        first_optimal_hour = min(self.optimal_posting_times)
        return next_day.replace(hour=first_optimal_hour, minute=0, second=0, microsecond=0)
        
    def increment_daily_count(self):
        """Increment the daily video count."""
        self.daily_video_count += 1
        self.logger.info(f"Daily video count updated: {self.daily_video_count}/{self.max_videos_per_day}")
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current scheduling statistics.
        
        Returns:
            Dict containing scheduling stats
        """
        return {
            'daily_video_count': self.daily_video_count,
            'max_videos_per_day': self.max_videos_per_day,
            'min_videos_per_day': self.min_videos_per_day,
            'last_reset_date': self.last_reset_date.isoformat(),
            'next_scheduled_time': self.get_next_scheduled_time().isoformat(),
            'optimal_posting_times': self.optimal_posting_times
        }