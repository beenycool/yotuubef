"""
YouTube Analytics Advisor
Fetches analytics data and uses Gemini AI to provide intelligent recommendations
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.integrations.youtube_client import YouTubeClient
from src.integrations.gemini_ai_client import GeminiAIClient
from src.management.channel_manager import ChannelManager
from src.analytics.analytics_feedback import AnalyticsFeedbackSystem
from src.config.settings import get_config
from src.utils.safe_print import safe_print

try:
    from google.genai import types
    TYPES_AVAILABLE = True
except ImportError:
    TYPES_AVAILABLE = False


class AnalyticsAdvisor:
    """
    YouTube Analytics Advisor that fetches analytics and provides AI-powered recommendations
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize clients
        self.youtube_client = YouTubeClient()
        self.gemini_client = GeminiAIClient()
        self.channel_manager = ChannelManager()
        self.feedback_system = AnalyticsFeedbackSystem()
        
        # Prompts for analytics analysis
        self.analytics_prompt = """
        Analyze this YouTube channel's analytics data and provide actionable recommendations:

        CHANNEL OVERVIEW:
        {channel_overview}

        VIDEO PERFORMANCE DATA:
        {video_analytics}

        RECENT UPLOADS:
        {recent_videos}

        Please provide analysis in the following format:

        ## 📊 ANALYTICS SUMMARY
        - Total views in last 30 days: [summarize]
        - Best performing video: [title and metrics]
        - Engagement trends: [improving/declining/stable]
        - Audience retention insights: [key findings]

        ## 🎯 KEY RECOMMENDATIONS
        1. **Content Strategy**: [specific content recommendations based on top performers]
        2. **Upload Timing**: [optimal timing based on performance patterns]
        3. **Thumbnail Optimization**: [specific suggestions for thumbnails]
        4. **Title Optimization**: [patterns from successful titles]
        5. **Retention Improvements**: [specific areas to focus on]

        ## ⚠️ IMMEDIATE ACTIONS
        - [3-5 specific actionable items to implement today]

        ## 📈 GROWTH OPPORTUNITIES
        - [Specific strategies to capitalize on current trends]
        - [Underperforming areas with highest improvement potential]

        ## 🎬 NEXT VIDEO SUGGESTIONS
        - [3 specific video ideas based on analytics patterns]
        - [Optimal video length based on retention data]
        - [Suggested posting schedule]

        Focus on data-driven, specific, and immediately actionable recommendations.
        """
        
        self.performance_comparison_prompt = """
        Compare this video's performance against channel averages and provide insights:

        VIDEO DETAILS:
        {video_details}

        CHANNEL AVERAGES:
        {channel_averages}

        PERFORMANCE COMPARISON:
        {performance_metrics}

        Provide specific insights on:
        1. Why this video performed above/below average
        2. What elements to replicate or avoid
        3. Specific improvements for future videos
        4. Thumbnail/title analysis if available
        """

    async def generate_startup_recommendations(self) -> Dict[str, Any]:
        """
        Generate comprehensive recommendations when main.py starts
        
        Returns:
            Dict containing analytics summary and recommendations
        """
        try:
            self.logger.info("Generating startup analytics recommendations...")
            
            # Add timeout protection to prevent hanging
            import asyncio
            
            # Fetch channel analytics data with timeout
            try:
                analytics_data = await asyncio.wait_for(
                    self._fetch_comprehensive_analytics(),
                    timeout=30.0  # 30 second timeout
                )
            except asyncio.TimeoutError:
                self.logger.warning("Analytics fetch timed out after 30 seconds")
                return self._generate_fallback_report("Analytics fetch timeout")
            
            if not analytics_data.get('success'):
                self.logger.warning(f"Analytics fetch failed: {analytics_data.get('error', 'Unknown error')}")
                return self._generate_fallback_report(analytics_data.get('error', 'Failed to fetch analytics data'))
            
            # Generate AI recommendations with timeout
            try:
                recommendations = await asyncio.wait_for(
                    self._generate_ai_recommendations(analytics_data),
                    timeout=20.0  # 20 second timeout
                )
            except asyncio.TimeoutError:
                self.logger.warning("AI recommendation generation timed out")
                recommendations = self._generate_fallback_recommendations(analytics_data)
            
            # Process recommendations through feedback system
            try:
                feedback_result = self.feedback_system.process_analytics_recommendations(
                    recommendations, analytics_data.get('summary', {})
                )
            except Exception as e:
                self.logger.warning(f"Feedback processing failed: {e}")
                feedback_result = {'insights': {}, 'optimization_params': {}}
            
            # Create startup report
            startup_report = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'analytics_summary': analytics_data.get('summary', {}),
                'recommendations': recommendations,
                'raw_analytics': analytics_data.get('raw_data', {}),
                'action_items': self._extract_action_items(recommendations),
                'feedback_insights': feedback_result.get('insights', {}),
                'optimization_params': feedback_result.get('optimization_params', {})
            }
            
            # Save report for reference (with timeout)
            try:
                await asyncio.wait_for(
                    self._save_analytics_report(startup_report),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                self.logger.warning("Report saving timed out")
            except Exception as e:
                self.logger.warning(f"Report saving failed: {e}")
            
            self.logger.info("Startup analytics recommendations generated successfully")
            return startup_report
            
        except Exception as e:
            self.logger.error(f"Failed to generate startup recommendations: {e}")
            return self._generate_fallback_report(str(e))

    async def _fetch_comprehensive_analytics(self) -> Dict[str, Any]:
        """Fetch comprehensive analytics data from YouTube"""
        try:
            # Get recent videos (last 30 days)
            recent_videos = await self.channel_manager.get_recent_videos(days=30)
            
            if not recent_videos:
                return {
                    'success': False,
                    'error': 'No recent videos found'
                }
            
            # Fetch analytics for each video
            video_analytics = []
            channel_totals = {
                'total_views': 0,
                'total_watch_time': 0,
                'total_subscribers_gained': 0,
                'total_videos': len(recent_videos)
            }
            
            for video in recent_videos[:20]:  # Limit to 20 most recent
                video_id = None  # Initialize video_id to avoid NameError
                try:
                    # Handle both old format {'id': {'videoId': 'xxx'}} and new format {'id': 'xxx'}
                    if isinstance(video.get('id'), dict):
                        video_id = video.get('id', {}).get('videoId')
                    else:
                        video_id = video.get('id') or video.get('video_id')
                    
                    if not video_id:
                        continue
                    
                    # Check if video is ready for analytics (>2 hours old)
                    if not self.channel_manager._is_video_ready_for_analytics(video):
                        self.logger.info(f"Skipping analytics for recent video: {video_id}")
                        continue
                    
                    analytics = await self.youtube_client.get_video_analytics(video_id)
                    
                    if analytics:
                        # Add video details
                        analytics['video_title'] = video.get('snippet', {}).get('title', 'Unknown')
                        analytics['published_at'] = video.get('snippet', {}).get('publishedAt')
                        analytics['video_id'] = video_id
                        
                        video_analytics.append(analytics)
                        
                        # Add to channel totals
                        channel_totals['total_views'] += analytics.get('views', 0)
                        channel_totals['total_watch_time'] += analytics.get('estimated_minutes_watched', 0)
                        channel_totals['total_subscribers_gained'] += analytics.get('subscribers_gained', 0)
                        
                except Exception as e:
                    # Use video_id safely - it's either None or a valid ID
                    video_identifier = video_id or "unknown video"
                    self.logger.warning(f"Failed to fetch analytics for video {video_identifier}: {e}")
                    continue
            
            if not video_analytics:
                return {
                    'success': False,
                    'error': 'No analytics data available for recent videos'
                }
            
            # Calculate averages
            num_videos = len(video_analytics)
            channel_averages = {
                'avg_views': channel_totals['total_views'] / num_videos if num_videos > 0 else 0,
                'avg_watch_time': channel_totals['total_watch_time'] / num_videos if num_videos > 0 else 0,
                'avg_subscribers_gained': channel_totals['total_subscribers_gained'] / num_videos if num_videos > 0 else 0,
            }
            
            # Identify top performers
            top_videos = sorted(video_analytics, key=lambda x: x.get('views', 0), reverse=True)[:5]
            
            return {
                'success': True,
                'summary': {
                    'analysis_period': '30 days',
                    'videos_analyzed': num_videos,
                    'channel_totals': channel_totals,
                    'channel_averages': channel_averages,
                    'top_performers': top_videos
                },
                'raw_data': {
                    'video_analytics': video_analytics,
                    'recent_videos': recent_videos
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to fetch comprehensive analytics: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _generate_ai_recommendations(self, analytics_data: Dict[str, Any]) -> Optional[str]:
        """Generate AI-powered recommendations using Gemini"""
        try:
            if not self.gemini_client.gemini_available:
                return self._generate_fallback_recommendations(analytics_data)
            
            # Prepare data for AI analysis
            summary = analytics_data.get('summary', {})
            raw_data = analytics_data.get('raw_data', {})
            
            # Format analytics data for the prompt
            channel_overview = json.dumps(summary, indent=2)
            video_analytics = json.dumps(raw_data.get('video_analytics', [])[:10], indent=2)  # Top 10 videos
            recent_videos = json.dumps([
                {
                    'title': v.get('snippet', {}).get('title', 'Unknown') if v.get('snippet') else v.get('title', 'Unknown'),
                    'published_at': v.get('snippet', {}).get('publishedAt') if v.get('snippet') else v.get('publishedAt'),
                    'video_id': (v.get('id', {}).get('videoId') if isinstance(v.get('id'), dict) else v.get('id')) or v.get('video_id')
                }
                for v in raw_data.get('recent_videos', [])[:10]
            ], indent=2)
            
            # Generate recommendations using Gemini
            prompt = self.analytics_prompt.format(
                channel_overview=channel_overview,
                video_analytics=video_analytics,
                recent_videos=recent_videos
            )
            
            # Rate limit and make API call
            await self.gemini_client.rate_limiter.wait_if_needed()
            
            response = await asyncio.to_thread(
                self.gemini_client.client.models.generate_content,
                model=self.gemini_client.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=3000,
                    candidate_count=1
                ) if TYPES_AVAILABLE else {
                    'temperature': 0.7,
                    'max_output_tokens': 3000,
                    'candidate_count': 1
                }
            )
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"AI recommendation generation failed: {e}")
            return self._generate_fallback_recommendations(analytics_data)

    def _generate_fallback_recommendations(self, analytics_data: Dict[str, Any]) -> str:
        """Generate basic recommendations when AI is not available"""
        try:
            summary = analytics_data.get('summary', {})
            totals = summary.get('channel_totals', {})
            averages = summary.get('channel_averages', {})
            top_videos = summary.get('top_performers', [])
            
            recommendations = f"""
## 📊 ANALYTICS SUMMARY
- Total views in last 30 days: {totals.get('total_views', 0):,}
- Videos analyzed: {summary.get('videos_analyzed', 0)}
- Average views per video: {averages.get('avg_views', 0):.0f}
- Total watch time: {totals.get('total_watch_time', 0):.0f} minutes
- Subscribers gained: {totals.get('total_subscribers_gained', 0)}

## 🎯 KEY RECOMMENDATIONS
1. **Content Strategy**: Focus on topics similar to your top-performing videos
2. **Upload Consistency**: Maintain regular upload schedule based on your current pattern
3. **Engagement**: Encourage more comments and interactions in your videos
4. **Thumbnails**: Create eye-catching thumbnails with clear focal points
5. **Titles**: Use engaging titles that clearly describe your content

## ⚠️ IMMEDIATE ACTIONS
- Review your top-performing video and identify what made it successful
- Check your video descriptions for optimization opportunities
- Ensure your thumbnails are mobile-friendly
- Respond to recent comments to boost engagement

## 📈 GROWTH OPPORTUNITIES
- Analyze your best-performing content themes and create more similar content
- Consider collaborating with other creators in your niche
- Optimize your video end screens to promote other videos

## 🎬 NEXT VIDEO SUGGESTIONS
- Create content similar to your top-performing videos
- Aim for video length that matches your average watch time
- Post during your historically best-performing times
"""
            
            if top_videos:
                top_video = top_videos[0]
                recommendations += f"\n### Top Performing Video:\n"
                recommendations += f"- Title: {top_video.get('video_title', 'N/A')}\n"
                recommendations += f"- Views: {top_video.get('views', 0):,}\n"
                recommendations += f"- Watch Time: {top_video.get('estimated_minutes_watched', 0):.0f} minutes\n"
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Fallback recommendations failed: {e}")
            return "Unable to generate recommendations. Please check your analytics data."

    def _generate_fallback_report(self, error_message: str) -> Dict[str, Any]:
        """Generate a fallback report when analytics fails"""
        return {
            'success': False,
            'timestamp': datetime.now().isoformat(),
            'error': error_message,
            'message': f'Analytics generation failed: {error_message}',
            'analytics_summary': {},
            'recommendations': self._generate_basic_fallback_recommendations(),
            'raw_analytics': {},
            'action_items': [
                "Check your YouTube API credentials",
                "Verify your channel has recent videos",
                "Ensure stable internet connection",
                "Try running the application again later"
            ],
            'feedback_insights': {},
            'optimization_params': {}
        }

    def _generate_basic_fallback_recommendations(self) -> str:
        """Generate very basic recommendations when everything fails"""
        return """
## 📊 ANALYTICS UNAVAILABLE
Unfortunately, analytics data could not be retrieved at this time.

## 🎯 GENERAL RECOMMENDATIONS
1. **Content Quality**: Focus on creating high-quality, engaging content
2. **Consistency**: Upload videos on a regular schedule
3. **SEO**: Use relevant keywords in titles and descriptions
4. **Thumbnails**: Create eye-catching, professional thumbnails
5. **Engagement**: Respond to comments and engage with your audience

## ⚠️ IMMEDIATE ACTIONS
- Check your YouTube API setup and credentials
- Ensure your channel has published videos
- Verify your internet connection
- Try running analytics again later

## 📈 GENERAL TIPS
- Study successful creators in your niche
- Keep up with YouTube trends and best practices
- Focus on your audience's interests and feedback
- Continuously improve your content quality
"""

    def _extract_action_items(self, recommendations: str) -> List[str]:
        """Extract actionable items from recommendations"""
        try:
            action_items = []
            
            # Handle None recommendations gracefully
            if not recommendations:
                return ["Review your analytics data", "Check your latest video performance", "Optimize your next video title"]
            
            # Look for immediate actions section
            if "IMMEDIATE ACTIONS" in recommendations:
                lines = recommendations.split('\n')
                in_actions_section = False
                
                for line in lines:
                    if "IMMEDIATE ACTIONS" in line:
                        in_actions_section = True
                        continue
                    elif in_actions_section and line.startswith('##'):
                        break
                    elif in_actions_section and line.strip().startswith('-'):
                        action_items.append(line.strip()[1:].strip())
            
            # If no specific section found, extract general action items
            if not action_items:
                lines = recommendations.split('\n')
                for line in lines:
                    if line.strip().startswith('- ') and any(word in line.lower() for word in ['should', 'need', 'must', 'check', 'review', 'create', 'optimize']):
                        action_items.append(line.strip()[2:])
            
            return action_items[:5]  # Limit to top 5 actions
            
        except Exception as e:
            self.logger.warning(f"Failed to extract action items: {e}")
            return ["Review your analytics data", "Check your latest video performance", "Optimize your next video title"]

    async def _save_analytics_report(self, report: Dict[str, Any]):
        """Save analytics report to file"""
        try:
            # Create analytics directory if it doesn't exist
            analytics_dir = Path("analytics_reports")
            analytics_dir.mkdir(exist_ok=True)
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analytics_report_{timestamp}.json"
            filepath = analytics_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Analytics report saved to {filepath}")
            # Keep only last N reports (configurable)
            reports = list(analytics_dir.glob("analytics_report_*.json"))
            limit = getattr(self.config, "analytics_report_limit", 30)
            if len(reports) > limit:
                reports.sort()
                for old_report in reports[:-limit]:
                    old_report.unlink()
            
            
        except Exception as e:
            self.logger.warning(f"Failed to save analytics report: {e}")

    async def analyze_specific_video(self, video_id: str) -> Dict[str, Any]:
        """Analyze a specific video's performance"""
        try:
            # Get video analytics
            analytics = await self.youtube_client.get_video_analytics(video_id)
            if not analytics:
                return {'success': False, 'error': 'Failed to fetch video analytics'}
            
            # Get channel averages for comparison
            channel_data = await self._fetch_comprehensive_analytics()
            channel_averages = channel_data.get('summary', {}).get('channel_averages', {})
            
            # Generate comparison analysis
            if self.gemini_client.gemini_available:
                comparison_analysis = await self._generate_video_comparison(
                    analytics, channel_averages
                )
            else:
                comparison_analysis = self._generate_basic_video_analysis(
                    analytics, channel_averages
                )
            
            return {
                'success': True,
                'video_analytics': analytics,
                'comparison_analysis': comparison_analysis,
                'channel_averages': channel_averages
            }
            
        except Exception as e:
            self.logger.error(f"Video analysis failed: {e}")
            return {'success': False, 'error': str(e)}

    async def _generate_video_comparison(self, video_analytics: Dict[str, Any], channel_averages: Dict[str, Any]) -> str:
        """Generate AI-powered video comparison analysis"""
        try:
            video_details = json.dumps(video_analytics, indent=2)
            averages = json.dumps(channel_averages, indent=2)
            
            # Calculate performance ratios
            performance_metrics = {
                'views_ratio': video_analytics.get('views', 0) / max(channel_averages.get('avg_views', 1), 1),
                'watch_time_ratio': video_analytics.get('estimated_minutes_watched', 0) / max(channel_averages.get('avg_watch_time', 1), 1),
                'subscribers_ratio': video_analytics.get('subscribers_gained', 0) / max(channel_averages.get('avg_subscribers_gained', 1), 1)
            }
            
            prompt = self.performance_comparison_prompt.format(
                video_details=video_details,
                channel_averages=averages,
                performance_metrics=json.dumps(performance_metrics, indent=2)
            )
            
            await self.gemini_client.rate_limiter.wait_if_needed()
            
            response = await asyncio.to_thread(
                self.gemini_client.client.models.generate_content,
                model=self.gemini_client.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.6,
                    max_output_tokens=2000,
                    candidate_count=1
                ) if TYPES_AVAILABLE else {
                    'temperature': 0.6,
                    'max_output_tokens': 2000,
                    'candidate_count': 1
                }
            )
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"AI video comparison failed: {e}")
            return self._generate_basic_video_analysis(video_analytics, channel_averages)

    def _generate_basic_video_analysis(self, video_analytics: Dict[str, Any], channel_averages: Dict[str, Any]) -> str:
        """Generate basic video analysis without AI"""
        try:
            views = video_analytics.get('views', 0)
            avg_views = channel_averages.get('avg_views', 0)
            
            performance = "above average" if views > avg_views else "below average"
            ratio = views / max(avg_views, 1) if avg_views > 0 else 1
            
            analysis = f"""
## Video Performance Analysis

**Performance**: This video is performing {performance} compared to your channel average.
- Video Views: {views:,}
- Channel Average: {avg_views:.0f}
- Performance Ratio: {ratio:.1f}x

**Watch Time**: {video_analytics.get('estimated_minutes_watched', 0):.0f} minutes
**Subscribers Gained**: {video_analytics.get('subscribers_gained', 0)}

**Recommendations**:
{'- This video is performing well! Consider creating similar content.' if ratio > 1.2 else '- Consider reviewing what made your top videos successful and apply those elements.'}
- Monitor engagement metrics and respond to comments promptly.
- Use this performance data to inform your next video strategy.
"""
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Basic video analysis failed: {e}")
            return "Unable to analyze video performance."

    def get_optimization_parameters_for_system(self) -> Dict[str, Any]:
        """
        Get optimization parameters for integration with the main system
        
        Returns:
            Optimization parameters that can be used by video processing
        """
        try:
            return self.feedback_system.get_optimization_parameters()
        except Exception as e:
            self.logger.error(f"Failed to get optimization parameters: {e}")
            return {}

    def apply_analytics_feedback_to_options(self, video_options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply analytics insights to video processing options
        
        Args:
            video_options: Current video processing options
            
        Returns:
            Enhanced options with analytics feedback applied
        """
        try:
            return self.feedback_system.apply_feedback_to_video_options(video_options)
        except Exception as e:
            self.logger.error(f"Failed to apply analytics feedback: {e}")
            return video_options

    def get_content_recommendations_for_finder(self) -> Dict[str, Any]:
        """
        Get content recommendations for the video finder system
        
        Returns:
            Content recommendations based on analytics
        """
        try:
            return self.feedback_system.get_content_recommendations()
        except Exception as e:
            self.logger.error(f"Failed to get content recommendations: {e}")
            return {}

    def track_video_performance_feedback(self, video_id: str, performance_data: Dict[str, Any]):
        """
        Track video performance for continuous improvement
        
        Args:
            video_id: YouTube video ID
            performance_data: Performance metrics
        """
        try:
            self.feedback_system.track_video_performance(video_id, performance_data)
        except Exception as e:
            self.logger.error(f"Failed to track video performance: {e}")

    def print_recommendations(self, report: Dict[str, Any]):
        """Print recommendations in a formatted way"""
        safe_print("\n" + "="*80)
        safe_print("🎯 YOUTUBE ANALYTICS ADVISOR - STARTUP RECOMMENDATIONS")
        safe_print("="*80)
        
        if not report.get('success'):
            safe_print(f"❌ Error: {report.get('error', 'Unknown error')}")
            safe_print(f"Message: {report.get('message', 'No additional information')}")
            return
        
        # Print summary
        summary = report.get('analytics_summary', {})
        if summary:
            totals = summary.get('channel_totals', {})
            safe_print(f"\n📊 QUICK STATS (Last 30 Days):")
            safe_print(f"   Views: {totals.get('total_views', 0):,}")
            safe_print(f"   Videos: {totals.get('total_videos', 0)}")
            safe_print(f"   Watch Time: {totals.get('total_watch_time', 0):.0f} minutes")
            safe_print(f"   Subscribers Gained: {totals.get('total_subscribers_gained', 0)}")
        
        # Print recommendations
        recommendations = report.get('recommendations', '')
        if recommendations:
            safe_print(f"\n{recommendations}")
        
        # Print action items
        action_items = report.get('action_items', [])
        if action_items:
            safe_print(f"\n⚡ TOP PRIORITY ACTIONS:")
            for i, action in enumerate(action_items, 1):
                safe_print(f"   {i}. {action}")
        
        # Print optimization insights if available
        optimization_params = report.get('optimization_params', {})
        if optimization_params:
            safe_print(f"\n🔧 SYSTEM OPTIMIZATION APPLIED:")
            
            # Content filtering
            content_filter = optimization_params.get('content_filtering', {})
            if content_filter.get('preferred_types'):
                safe_print(f"   Preferred content types: {', '.join(content_filter['preferred_types'])}")
            
            # Title optimization
            title_opt = optimization_params.get('title_optimization', {})
            enabled_title_opts = [k for k, v in title_opt.items() if v]
            if enabled_title_opts:
                safe_print(f"   Title optimization: {', '.join(enabled_title_opts)}")
            
            # Processing preferences
            proc_pref = optimization_params.get('processing_preferences', {})
            enabled_prefs = [k for k, v in proc_pref.items() if v]
            if enabled_prefs:
                safe_print(f"   Processing enhancements: {', '.join(enabled_prefs)}")
        
        safe_print("\n" + "="*80)
        safe_print(f"📝 Full report saved to analytics_reports/")
        safe_print(f"🔄 Feedback applied to future video processing")
        safe_print("="*80 + "\n")