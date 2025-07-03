"""
Analytics Feedback System
Feeds analytics insights back into the main YouTube processing system for continuous improvement
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

from src.config.settings import get_config


class AnalyticsFeedbackSystem:
    """
    System that feeds analytics insights back into video processing for continuous improvement
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Feedback storage
        self.feedback_file = Path("data/analytics_feedback.json")
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing feedback data
        self.feedback_data = self._load_feedback_data()
        
    def _load_feedback_data(self) -> Dict[str, Any]:
        """Load existing feedback data"""
        try:
            if self.feedback_file.exists():
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return self._create_default_feedback_data()
        except Exception as e:
            self.logger.warning(f"Failed to load feedback data: {e}")
            return self._create_default_feedback_data()
    
    def _create_default_feedback_data(self) -> Dict[str, Any]:
        """Create default feedback data structure"""
        return {
            'content_preferences': {
                'successful_keywords': [],
                'successful_topics': [],
                'optimal_length_range': [30, 60],
                'best_posting_times': [],
                'successful_thumbnail_styles': []
            },
            'optimization_settings': {
                'title_optimization_enabled': True,
                'thumbnail_optimization_enabled': True,
                'content_filtering_enabled': True,
                'timing_optimization_enabled': True
            },
            'performance_patterns': {
                'high_performing_subreddits': [],
                'successful_content_types': [],
                'optimal_video_characteristics': {}
            },
            'ai_insights': {
                'recent_recommendations': [],
                'applied_changes': [],
                'success_metrics': {}
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_feedback_data(self):
        """Save feedback data to file"""
        try:
            self.feedback_data['last_updated'] = datetime.now().isoformat()
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, indent=2, ensure_ascii=False, default=str)
            self.logger.debug("Feedback data saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save feedback data: {e}")
    
    def process_analytics_recommendations(self, recommendations: str, analytics_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process analytics recommendations and extract actionable insights
        
        Args:
            recommendations: AI-generated recommendations text
            analytics_summary: Summary of analytics data
            
        Returns:
            Processed insights for system integration
        """
        try:
            insights = {
                'content_insights': self._extract_content_insights(recommendations),
                'timing_insights': self._extract_timing_insights(recommendations),
                'optimization_insights': self._extract_optimization_insights(recommendations),
                'performance_patterns': self._analyze_performance_patterns(analytics_summary)
            }
            
            # Update feedback data with new insights
            self._update_feedback_with_insights(insights)
            
            # Generate optimization parameters
            optimization_params = self._generate_optimization_parameters(insights)
            
            self.logger.info("Analytics recommendations processed successfully")
            return {
                'success': True,
                'insights': insights,
                'optimization_params': optimization_params,
                'feedback_updated': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process analytics recommendations: {e}")
            return {'success': False, 'error': str(e)}
    
    def _parse_ai_feedback(self, recommendations: str) -> Optional[Dict[str, Any]]:
        """
        Try to parse AI feedback as JSON, fallback to regex with named groups if needed.
        """
        if not recommendations:
            return None
        # Try JSON first
        try:
            return json.loads(recommendations)
        except Exception:
            pass
        # Fallback: regex for key fields
        pattern = re.compile(
            r'"?(?P<key>\w+)"?\s*:\s*(?P<value>\[.*?\]|".*?"|\d+|true|false|null)',
            re.DOTALL | re.IGNORECASE
        )
        result = {}
        for m in pattern.finditer(recommendations):
            key = m.group("key")
            value = m.group("value")
            try:
                value = json.loads(value)
            except Exception:
                value = value.strip('"')
            result[key] = value
        return result if result else None

    def _extract_content_insights(self, recommendations: str) -> Dict[str, Any]:
        """Extract content-related insights from recommendations"""
        # Try to parse as structured data
        parsed = self._parse_ai_feedback(recommendations)
        if parsed and "content_insights" in parsed:
            return parsed["content_insights"]
        content_insights = {
            'preferred_keywords': [],
            'successful_topics': [],
            'content_types': [],
            'length_recommendations': {}
        }
        if not recommendations:
            return content_insights
        rec_lower = recommendations.lower()
        if 'tutorial' in rec_lower or 'how to' in rec_lower:
            content_insights['content_types'].append('tutorial')
        if 'entertainment' in rec_lower:
            content_insights['content_types'].append('entertainment')
        if 'educational' in rec_lower:
            content_insights['content_types'].append('educational')
        if 'comedy' in rec_lower or 'funny' in rec_lower:
            content_insights['content_types'].append('comedy')
        if 'short' in rec_lower and ('60' in rec_lower or 'minute' in rec_lower):
            content_insights['length_recommendations']['optimal_max'] = 60
        if '30' in rec_lower and 'second' in rec_lower:
            content_insights['length_recommendations']['optimal_min'] = 30
        if 'numbers' in rec_lower:
            content_insights['preferred_keywords'].extend(['numbers', 'list_format'])
        if 'amazing' in rec_lower or 'incredible' in rec_lower:
            content_insights['preferred_keywords'].extend(['amazing', 'incredible'])
        return content_insights
    
    def _extract_timing_insights(self, recommendations: str) -> Dict[str, Any]:
        """Extract timing-related insights from recommendations"""
        parsed = self._parse_ai_feedback(recommendations)
        if parsed and "timing_insights" in parsed:
            return parsed["timing_insights"]
        timing_insights = {
            'optimal_days': [],
            'optimal_hours': [],
            'posting_frequency': {}
        }
        if not recommendations:
            return timing_insights
        rec_lower = recommendations.lower()
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for day in days:
            if day in rec_lower:
                timing_insights['optimal_days'].append(day)
        if 'morning' in rec_lower:
            timing_insights['optimal_hours'].extend([9, 10, 11])
        if 'afternoon' in rec_lower:
            timing_insights['optimal_hours'].extend([14, 15, 16])
        if 'evening' in rec_lower:
            timing_insights['optimal_hours'].extend([19, 20, 21])
        time_pattern = r'(?P<hour>\d{1,2})[-\s]*(\d{1,2})?\s*(?P<ampm>pm|am)'
        for match in re.finditer(time_pattern, rec_lower):
            hour_start = int(match.group("hour"))
            if match.group("ampm") == 'pm' and hour_start != 12:
                hour_start += 12
            timing_insights['optimal_hours'].append(hour_start)
        return timing_insights
    
    def _extract_optimization_insights(self, recommendations: str) -> Dict[str, Any]:
        """Extract optimization-related insights from recommendations"""
        parsed = self._parse_ai_feedback(recommendations)
        if parsed and "optimization_insights" in parsed:
            return parsed["optimization_insights"]
        optimization_insights = {
            'title_strategies': [],
            'thumbnail_strategies': [],
            'engagement_strategies': []
        }
        if not recommendations:
            return optimization_insights
        rec_lower = recommendations.lower()
        if 'title' in rec_lower:
            if 'number' in rec_lower:
                optimization_insights['title_strategies'].append('include_numbers')
            if 'how to' in rec_lower:
                optimization_insights['title_strategies'].append('how_to_format')
            if 'question' in rec_lower:
                optimization_insights['title_strategies'].append('question_format')
        if 'thumbnail' in rec_lower:
            if 'contrast' in rec_lower:
                optimization_insights['thumbnail_strategies'].append('high_contrast')
            if 'text' in rec_lower:
                optimization_insights['thumbnail_strategies'].append('include_text')
            if 'face' in rec_lower:
                optimization_insights['thumbnail_strategies'].append('include_face')
        if 'hook' in rec_lower:
            optimization_insights['engagement_strategies'].append('strong_opening_hook')
        if 'caption' in rec_lower:
            optimization_insights['engagement_strategies'].append('add_captions')
        if 'end screen' in rec_lower:
            optimization_insights['engagement_strategies'].append('optimize_end_screen')
        return optimization_insights
    
    def _analyze_performance_patterns(self, analytics_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance patterns from analytics data"""
        patterns = {
            'top_performing_characteristics': {},
            'underperforming_patterns': {},
            'growth_trends': {}
        }
        
        try:
            # Analyze top performers
            top_performers = analytics_summary.get('top_performers', [])
            if top_performers:
                top_video = top_performers[0]
                patterns['top_performing_characteristics'] = {
                    'title_length': len(top_video.get('video_title', '')),
                    'high_performance_threshold': top_video.get('views', 0),
                    'successful_characteristics': []
                }
            
            # Analyze channel averages and trends
            channel_totals = analytics_summary.get('channel_totals', {})
            patterns['growth_trends'] = {
                'total_views': channel_totals.get('total_views', 0),
                'subscriber_growth': channel_totals.get('total_subscribers_gained', 0),
                'average_performance': analytics_summary.get('channel_averages', {})
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze performance patterns: {e}")
        
        return patterns
    
    def _update_feedback_with_insights(self, insights: Dict[str, Any]):
        """Update feedback data with new insights"""
        try:
            # Update content preferences
            content = insights.get('content_insights', {})
            self.feedback_data['content_preferences']['successful_topics'].extend(
                content.get('content_types', [])
            )
            
            # Update optimization settings based on insights
            optimization = insights.get('optimization_insights', {})
            if optimization.get('title_strategies'):
                self.feedback_data['optimization_settings']['title_optimization_enabled'] = True
            if optimization.get('thumbnail_strategies'):
                self.feedback_data['optimization_settings']['thumbnail_optimization_enabled'] = True
            
            # Update performance patterns
            patterns = insights.get('performance_patterns', {})
            if patterns:
                self.feedback_data['performance_patterns']['optimal_video_characteristics'].update(
                    patterns.get('top_performing_characteristics', {})
                )
            
            # Remove duplicates and keep only recent data
            self._clean_feedback_data()
            
            # Save updated data
            self._save_feedback_data()
            
        except Exception as e:
            self.logger.error(f"Failed to update feedback data: {e}")
    
    def _clean_feedback_data(self):
        """Clean and optimize feedback data"""
        try:
            # Remove duplicates from lists
            for key in ['successful_topics', 'successful_keywords']:
                if key in self.feedback_data['content_preferences']:
                    items = self.feedback_data['content_preferences'][key]
                    self.feedback_data['content_preferences'][key] = list(set(items))[-self.config.successful_topics_limit:]  # Configurable limit
            
            # Limit recent recommendations
            recent_recs = self.feedback_data['ai_insights'].get('recent_recommendations', [])
            self.feedback_data['ai_insights']['recent_recommendations'] = recent_recs[-self.config.recent_recommendations_limit:]  # Configurable limit
            
        except Exception as e:
            self.logger.warning(f"Failed to clean feedback data: {e}")
    
    def _generate_optimization_parameters(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization parameters for the main system"""
        params = {
            'content_filtering': {},
            'title_optimization': {},
            'thumbnail_optimization': {},
            'timing_optimization': {},
            'processing_preferences': {}
        }
        
        try:
            # Content filtering parameters
            content = insights.get('content_insights', {})
            if content.get('content_types'):
                params['content_filtering']['preferred_types'] = content['content_types']
            if content.get('length_recommendations'):
                params['content_filtering']['length_preferences'] = content['length_recommendations']
            
            # Title optimization parameters
            title_strategies = insights.get('optimization_insights', {}).get('title_strategies', [])
            params['title_optimization'] = {
                'include_numbers': 'include_numbers' in title_strategies,
                'use_how_to_format': 'how_to_format' in title_strategies,
                'use_question_format': 'question_format' in title_strategies
            }
            
            # Thumbnail optimization parameters
            thumb_strategies = insights.get('optimization_insights', {}).get('thumbnail_strategies', [])
            params['thumbnail_optimization'] = {
                'use_high_contrast': 'high_contrast' in thumb_strategies,
                'include_text_overlay': 'include_text' in thumb_strategies,
                'include_faces': 'include_face' in thumb_strategies
            }
            
            # Timing optimization parameters
            timing = insights.get('timing_insights', {})
            params['timing_optimization'] = {
                'optimal_days': timing.get('optimal_days', []),
                'optimal_hours': timing.get('optimal_hours', []),
                'scheduling_enabled': len(timing.get('optimal_days', [])) > 0
            }
            
            # Processing preferences
            engagement_strategies = insights.get('optimization_insights', {}).get('engagement_strategies', [])
            params['processing_preferences'] = {
                'add_strong_hooks': 'strong_opening_hook' in engagement_strategies,
                'add_captions': 'add_captions' in engagement_strategies,
                'optimize_end_screens': 'optimize_end_screen' in engagement_strategies
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate optimization parameters: {e}")
        
        return params
    
    def get_optimization_parameters(self) -> Dict[str, Any]:
        """Get current optimization parameters for system integration"""
        try:
            # Generate parameters from stored feedback data
            mock_insights = {
                'content_insights': {
                    'content_types': self.feedback_data['content_preferences'].get('successful_topics', []),
                    'length_recommendations': {}
                },
                'optimization_insights': {
                    'title_strategies': [],
                    'thumbnail_strategies': [],
                    'engagement_strategies': []
                },
                'timing_insights': {
                    'optimal_days': self.feedback_data['content_preferences'].get('best_posting_times', []),
                    'optimal_hours': []
                }
            }
            
            return self._generate_optimization_parameters(mock_insights)
            
        except Exception as e:
            self.logger.error(f"Failed to get optimization parameters: {e}")
            return {}
    
    def apply_feedback_to_video_options(self, video_options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply analytics feedback to video processing options
        
        Args:
            video_options: Current video processing options
            
        Returns:
            Enhanced video options with analytics feedback applied
        """
        try:
            enhanced_options = video_options.copy()
            optimization_params = self.get_optimization_parameters()
            
            # Apply content filtering preferences
            content_filter = optimization_params.get('content_filtering', {})
            if content_filter.get('preferred_types'):
                enhanced_options['preferred_content_types'] = content_filter['preferred_types']
            
            # Apply title optimization
            title_opt = optimization_params.get('title_optimization', {})
            enhanced_options.update({
                'optimize_titles': True,
                'title_include_numbers': title_opt.get('include_numbers', False),
                'title_use_how_to': title_opt.get('use_how_to_format', False),
                'title_use_questions': title_opt.get('use_question_format', False)
            })
            
            # Apply thumbnail optimization
            thumb_opt = optimization_params.get('thumbnail_optimization', {})
            enhanced_options.update({
                'thumbnail_high_contrast': thumb_opt.get('use_high_contrast', True),
                'thumbnail_include_text': thumb_opt.get('include_text_overlay', True),
                'thumbnail_include_faces': thumb_opt.get('include_faces', False)
            })
            
            # Apply processing preferences
            proc_pref = optimization_params.get('processing_preferences', {})
            enhanced_options.update({
                'add_opening_hooks': proc_pref.get('add_strong_hooks', True),
                'add_captions': proc_pref.get('add_captions', True),
                'optimize_end_screens': proc_pref.get('optimize_end_screens', True)
            })
            
            self.logger.info("Applied analytics feedback to video options")
            return enhanced_options
            
        except Exception as e:
            self.logger.error(f"Failed to apply feedback to video options: {e}")
            return video_options
    
    def get_content_recommendations(self) -> Dict[str, Any]:
        """Get content recommendations based on analytics feedback"""
        try:
            recommendations = {
                'preferred_subreddits': self.feedback_data['performance_patterns'].get('high_performing_subreddits', []),
                'successful_content_types': self.feedback_data['content_preferences'].get('successful_topics', []),
                'optimal_video_length': self.feedback_data['content_preferences'].get('optimal_length_range', [30, 60]),
                'best_posting_times': self.feedback_data['content_preferences'].get('best_posting_times', []),
                'successful_keywords': self.feedback_data['content_preferences'].get('successful_keywords', [])
            }
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to get content recommendations: {e}")
            return {}
    
    def track_video_performance(self, video_id: str, performance_data: Dict[str, Any]):
        """Track video performance to improve future recommendations"""
        try:
            # Store performance data for analysis
            if 'tracked_videos' not in self.feedback_data:
                self.feedback_data['tracked_videos'] = {}
            
            self.feedback_data['tracked_videos'][video_id] = {
                'performance': performance_data,
                'tracked_at': datetime.now().isoformat()
            }
            
            # Keep only recent tracked videos (last 50)
            tracked = self.feedback_data['tracked_videos']
            if len(tracked) > 50:
                # Keep most recent 50
                sorted_videos = sorted(tracked.items(), key=lambda x: x[1]['tracked_at'], reverse=True)
                self.feedback_data['tracked_videos'] = dict(sorted_videos[:50])
            
            self._save_feedback_data()
            self.logger.info(f"Tracked performance for video {video_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to track video performance: {e}")