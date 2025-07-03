"""
Engagement Metrics Monitoring System for YouTube video enhancement impact analysis.
Tracks performance data, A/B testing, and enhancement effectiveness.
"""

import logging
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

from src.config.settings import get_config


class EnhancementType(Enum):
    """Types of video enhancements"""
    SOUND_EFFECTS = "sound_effects"
    VISUAL_EFFECTS = "visual_effects"
    TEXT_OVERLAYS = "text_overlays"
    COLOR_GRADING = "color_grading"
    DYNAMIC_ZOOM = "dynamic_zoom"
    BACKGROUND_MUSIC = "background_music"
    SPEED_EFFECTS = "speed_effects"
    HOOK_TEXT = "hook_text"


@dataclass
class VideoMetrics:
    """Video performance metrics"""
    video_id: str
    title: str
    upload_date: datetime
    duration_seconds: float
    
    # Engagement metrics
    views: int = 0
    likes: int = 0
    dislikes: int = 0
    comments: int = 0
    shares: int = 0
    watch_time_seconds: float = 0.0
    retention_rate: float = 0.0
    click_through_rate: float = 0.0
    
    # Enhancement data
    enhancements_used: List[str] = None
    sound_effects_count: int = 0
    visual_effects_count: int = 0
    
    # Calculated metrics
    engagement_rate: float = 0.0
    likes_ratio: float = 0.0
    completion_rate: float = 0.0
    
    def __post_init__(self):
        if self.enhancements_used is None:
            self.enhancements_used = []
        
        # Calculate derived metrics
        self._calculate_derived_metrics()
    
    def _calculate_derived_metrics(self):
        """Calculate derived engagement metrics"""
        if self.views > 0:
            total_engagement = self.likes + self.comments + self.shares
            self.engagement_rate = (total_engagement / self.views) * 100
            
            total_reactions = self.likes + self.dislikes
            if total_reactions > 0:
                self.likes_ratio = (self.likes / total_reactions) * 100
        
        if self.duration_seconds > 0 and self.watch_time_seconds > 0:
            self.completion_rate = (self.watch_time_seconds / self.duration_seconds) * 100


@dataclass
class EnhancementPerformance:
    """Performance analysis for specific enhancements"""
    enhancement_type: str
    videos_with_enhancement: int
    videos_without_enhancement: int
    
    # Average metrics with enhancement
    avg_views_with: float = 0.0
    avg_engagement_with: float = 0.0
    avg_retention_with: float = 0.0
    avg_completion_with: float = 0.0
    
    # Average metrics without enhancement
    avg_views_without: float = 0.0
    avg_engagement_without: float = 0.0
    avg_retention_without: float = 0.0
    avg_completion_without: float = 0.0
    
    # Performance improvements
    views_improvement: float = 0.0
    engagement_improvement: float = 0.0
    retention_improvement: float = 0.0
    completion_improvement: float = 0.0
    
    # Statistical significance
    confidence_level: float = 0.0
    sample_size_adequate: bool = False


class EngagementMetricsDB:
    """Database manager for engagement metrics"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        if db_path:
            self.db_path = db_path
        else:
            self.db_path = self.config.paths.base_dir / "data" / "databases" / "engagement_metrics.db"
        
        self._init_database()
    
    def _init_database(self):
        """Initialize the metrics database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Video metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS video_metrics (
                        video_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        upload_date TEXT NOT NULL,
                        duration_seconds REAL NOT NULL,
                        views INTEGER DEFAULT 0,
                        likes INTEGER DEFAULT 0,
                        dislikes INTEGER DEFAULT 0,
                        comments INTEGER DEFAULT 0,
                        shares INTEGER DEFAULT 0,
                        watch_time_seconds REAL DEFAULT 0,
                        retention_rate REAL DEFAULT 0,
                        click_through_rate REAL DEFAULT 0,
                        enhancements_used TEXT,
                        sound_effects_count INTEGER DEFAULT 0,
                        visual_effects_count INTEGER DEFAULT 0,
                        engagement_rate REAL DEFAULT 0,
                        likes_ratio REAL DEFAULT 0,
                        completion_rate REAL DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Enhancement tracking table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS enhancement_tracking (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        video_id TEXT,
                        enhancement_type TEXT,
                        parameters TEXT,
                        applied_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (video_id) REFERENCES video_metrics (video_id)
                    )
                """)
                
                # Performance snapshots table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        video_id TEXT,
                        snapshot_date TEXT,
                        views INTEGER,
                        likes INTEGER,
                        dislikes INTEGER,
                        comments INTEGER,
                        shares INTEGER,
                        watch_time_seconds REAL,
                        retention_rate REAL,
                        FOREIGN KEY (video_id) REFERENCES video_metrics (video_id)
                    )
                """)
                
                # A/B test results table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ab_test_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        test_name TEXT,
                        video_id TEXT,
                        variant TEXT,
                        enhancement_config TEXT,
                        performance_metrics TEXT,
                        test_start_date TEXT,
                        test_end_date TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                self.logger.info("Engagement metrics database initialized")
                
        except Exception as e:
            self.logger.error(f"Error initializing engagement metrics database: {e}")
    
    def store_video_metrics(self, metrics: VideoMetrics) -> bool:
        """Store or update video metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                enhancements_json = json.dumps(metrics.enhancements_used)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO video_metrics (
                        video_id, title, upload_date, duration_seconds,
                        views, likes, dislikes, comments, shares,
                        watch_time_seconds, retention_rate, click_through_rate,
                        enhancements_used, sound_effects_count, visual_effects_count,
                        engagement_rate, likes_ratio, completion_rate, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.video_id, metrics.title, metrics.upload_date.isoformat(),
                    metrics.duration_seconds, metrics.views, metrics.likes,
                    metrics.dislikes, metrics.comments, metrics.shares,
                    metrics.watch_time_seconds, metrics.retention_rate,
                    metrics.click_through_rate, enhancements_json,
                    metrics.sound_effects_count, metrics.visual_effects_count,
                    metrics.engagement_rate, metrics.likes_ratio,
                    metrics.completion_rate, datetime.now().isoformat()
                ))
                
                conn.commit()
                self.logger.debug(f"Stored metrics for video {metrics.video_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error storing video metrics: {e}")
            return False
    
    def get_video_metrics(self, video_id: str) -> Optional[VideoMetrics]:
        """Retrieve video metrics by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM video_metrics WHERE video_id = ?
                """, (video_id,))
                
                row = cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    data = dict(zip(columns, row))
                    
                    # Parse JSON fields
                    data['enhancements_used'] = json.loads(data['enhancements_used'] or '[]')
                    data['upload_date'] = datetime.fromisoformat(data['upload_date'])
                    
                    # Remove database-specific fields
                    data.pop('created_at', None)
                    data.pop('updated_at', None)
                    
                    return VideoMetrics(**data)
                
        except Exception as e:
            self.logger.error(f"Error retrieving video metrics: {e}")
        
        return None
    
    def track_enhancement(self, video_id: str, enhancement_type: str, parameters: Dict[str, Any]):
        """Track applied enhancement"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO enhancement_tracking (video_id, enhancement_type, parameters)
                    VALUES (?, ?, ?)
                """, (video_id, enhancement_type, json.dumps(parameters)))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error tracking enhancement: {e}")


class EngagementAnalyzer:
    """Analyzes engagement metrics and enhancement effectiveness"""
    
    def __init__(self, db: EngagementMetricsDB):
        self.db = db
        self.logger = logging.getLogger(__name__)
    
    def analyze_enhancement_performance(self, enhancement_type: str, 
                                       days_back: int = 30) -> Optional[EnhancementPerformance]:
        """Analyze performance of specific enhancement type"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # Get videos with the enhancement
                cursor.execute("""
                    SELECT vm.* FROM video_metrics vm
                    WHERE vm.upload_date >= ? 
                    AND vm.enhancements_used LIKE ?
                """, (cutoff_date, f'%{enhancement_type}%'))
                
                with_enhancement = cursor.fetchall()
                
                # Get videos without the enhancement
                cursor.execute("""
                    SELECT vm.* FROM video_metrics vm
                    WHERE vm.upload_date >= ? 
                    AND (vm.enhancements_used NOT LIKE ? OR vm.enhancements_used IS NULL)
                """, (cutoff_date, f'%{enhancement_type}%'))
                
                without_enhancement = cursor.fetchall()
                
                if not with_enhancement or not without_enhancement:
                    self.logger.warning(f"Insufficient data for {enhancement_type} analysis")
                    return None
                
                # Calculate averages
                columns = [desc[0] for desc in cursor.description]
                
                with_metrics = [dict(zip(columns, row)) for row in with_enhancement]
                without_metrics = [dict(zip(columns, row)) for row in without_enhancement]
                
                performance = EnhancementPerformance(
                    enhancement_type=enhancement_type,
                    videos_with_enhancement=len(with_metrics),
                    videos_without_enhancement=len(without_metrics)
                )
                
                # Calculate averages for videos with enhancement
                if with_metrics:
                    performance.avg_views_with = statistics.mean([m['views'] for m in with_metrics])
                    performance.avg_engagement_with = statistics.mean([m['engagement_rate'] for m in with_metrics])
                    performance.avg_retention_with = statistics.mean([m['retention_rate'] for m in with_metrics])
                    performance.avg_completion_with = statistics.mean([m['completion_rate'] for m in with_metrics])
                
                # Calculate averages for videos without enhancement
                if without_metrics:
                    performance.avg_views_without = statistics.mean([m['views'] for m in without_metrics])
                    performance.avg_engagement_without = statistics.mean([m['engagement_rate'] for m in without_metrics])
                    performance.avg_retention_without = statistics.mean([m['retention_rate'] for m in without_metrics])
                    performance.avg_completion_without = statistics.mean([m['completion_rate'] for m in without_metrics])
                
                # Calculate improvements
                performance.views_improvement = self._calculate_improvement(
                    performance.avg_views_with, performance.avg_views_without
                )
                performance.engagement_improvement = self._calculate_improvement(
                    performance.avg_engagement_with, performance.avg_engagement_without
                )
                performance.retention_improvement = self._calculate_improvement(
                    performance.avg_retention_with, performance.avg_retention_without
                )
                performance.completion_improvement = self._calculate_improvement(
                    performance.avg_completion_with, performance.avg_completion_without
                )
                
                # Check sample size adequacy (simplified)
                performance.sample_size_adequate = (
                    len(with_metrics) >= 10 and len(without_metrics) >= 10
                )
                
                return performance
                
        except Exception as e:
            self.logger.error(f"Error analyzing enhancement performance: {e}")
            return None
    
    def _calculate_improvement(self, with_value: float, without_value: float) -> float:
        """Calculate percentage improvement"""
        if without_value == 0:
            return 0.0
        return ((with_value - without_value) / without_value) * 100
    
    def get_top_performing_enhancements(self, days_back: int = 30) -> List[Tuple[str, float]]:
        """Get top performing enhancements by engagement improvement"""
        enhancements = []
        
        for enhancement_type in EnhancementType:
            performance = self.analyze_enhancement_performance(enhancement_type.value, days_back)
            if performance and performance.sample_size_adequate:
                enhancements.append((enhancement_type.value, performance.engagement_improvement))
        
        # Sort by engagement improvement
        enhancements.sort(key=lambda x: x[1], reverse=True)
        return enhancements
    
    def generate_performance_report(self, days_back: int = 30) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            report = {
                'report_date': datetime.now().isoformat(),
                'analysis_period_days': days_back,
                'enhancement_performance': {},
                'summary': {
                    'total_videos_analyzed': 0,
                    'most_effective_enhancement': None,
                    'average_improvement': 0.0,
                    'recommendations': []
                }
            }
            
            all_improvements = []
            total_videos = 0
            
            for enhancement_type in EnhancementType:
                performance = self.analyze_enhancement_performance(enhancement_type.value, days_back)
                if performance:
                    report['enhancement_performance'][enhancement_type.value] = asdict(performance)
                    total_videos += performance.videos_with_enhancement + performance.videos_without_enhancement
                    
                    if performance.sample_size_adequate:
                        all_improvements.append(performance.engagement_improvement)
            
            # Calculate summary statistics
            report['summary']['total_videos_analyzed'] = total_videos
            
            if all_improvements:
                report['summary']['average_improvement'] = statistics.mean(all_improvements)
                
                # Find most effective enhancement
                top_enhancements = self.get_top_performing_enhancements(days_back)
                if top_enhancements:
                    report['summary']['most_effective_enhancement'] = top_enhancements[0][0]
                    
                    # Generate recommendations
                    recommendations = []
                    for enhancement, improvement in top_enhancements[:3]:
                        if improvement > 5:  # 5% improvement threshold
                            recommendations.append(f"Increase use of {enhancement} (showing {improvement:.1f}% improvement)")
                    
                    report['summary']['recommendations'] = recommendations
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {}


class EngagementMonitor:
    """Main engagement monitoring system"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.db = EngagementMetricsDB()
        self.analyzer = EngagementAnalyzer(self.db)
    
    def record_video_upload(self, video_id: str, title: str, duration: float, 
                           enhancements: List[str]) -> bool:
        """Record a new video upload with its enhancements"""
        metrics = VideoMetrics(
            video_id=video_id,
            title=title,
            upload_date=datetime.now(),
            duration_seconds=duration,
            enhancements_used=enhancements,
            sound_effects_count=enhancements.count('sound_effects'),
            visual_effects_count=sum(1 for e in enhancements if 'visual' in e)
        )
        
        return self.db.store_video_metrics(metrics)
    
    def update_metrics_from_youtube(self, video_id: str, youtube_data: Dict[str, Any]) -> bool:
        """Update metrics with data from YouTube API"""
        try:
            # Get existing metrics
            metrics = self.db.get_video_metrics(video_id)
            if not metrics:
                self.logger.warning(f"No existing metrics found for video {video_id}")
                return False
            
            # Update with YouTube data
            statistics_data = youtube_data.get('statistics', {})
            metrics.views = int(statistics_data.get('viewCount', 0))
            metrics.likes = int(statistics_data.get('likeCount', 0))
            metrics.dislikes = int(statistics_data.get('dislikeCount', 0))
            metrics.comments = int(statistics_data.get('commentCount', 0))
            
            # Recalculate derived metrics
            metrics._calculate_derived_metrics()
            
            return self.db.store_video_metrics(metrics)
            
        except Exception as e:
            self.logger.error(f"Error updating metrics from YouTube: {e}")
            return False
    
    def get_enhancement_recommendations(self) -> List[str]:
        """Get recommendations for which enhancements to use"""
        top_enhancements = self.analyzer.get_top_performing_enhancements()
        
        recommendations = []
        for enhancement, improvement in top_enhancements[:3]:
            if improvement > 5:  # 5% improvement threshold
                recommendations.append(enhancement)
        
        return recommendations
    
    def export_performance_report(self, output_path: Optional[Path] = None) -> Path:
        """Export performance report to JSON file"""
        report = self.analyzer.generate_performance_report()
        
        if not output_path:
            output_path = self.config.paths.base_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Performance report exported to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting performance report: {e}")
            raise