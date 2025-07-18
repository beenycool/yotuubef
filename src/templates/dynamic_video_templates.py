"""
Dynamic Video Templates System
Provides multiple video formats with intelligent selection and automated asset sourcing
"""

import asyncio
import logging
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import re

# Optional dependencies with fallbacks
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from src.config.settings import get_config


class VideoTemplateType(Enum):
    """Available video template types"""
    LIST_STYLE = "list_style"
    SINGLE_STORY = "single_story"
    QA_FORMAT = "qa_format"
    TUTORIAL = "tutorial"
    NEWS_STYLE = "news_style"
    COMPILATION = "compilation"


@dataclass
class VideoAsset:
    """Represents a video asset (image, video, audio)"""
    url: str
    asset_type: str  # 'image', 'video', 'audio'
    description: str
    keywords: List[str]
    source: str
    license_type: str = "free"
    duration: Optional[float] = None
    resolution: Optional[Tuple[int, int]] = None


@dataclass
class VideoTemplate:
    """Video template configuration"""
    template_type: VideoTemplateType
    name: str
    description: str
    structure: List[Dict[str, Any]]  # Template structure
    asset_requirements: Dict[str, int]  # Asset requirements per type
    duration_range: Tuple[int, int]  # Min/max duration in seconds
    style_config: Dict[str, Any]  # Visual and audio style configuration
    compatibility_score: float  # How well it works with different content


class DynamicVideoTemplateManager:
    """
    Manages dynamic video templates with intelligent selection and automated asset sourcing
    """
    
    def __init__(self):
        try:
            self.config = get_config()
        except:
            # Fallback for testing
            self.config = {
                'PEXELS_API_KEY': '',
                'UNSPLASH_API_KEY': '',
                'PIXABAY_API_KEY': ''
            }
        self.logger = logging.getLogger(__name__)
        
        # Asset sources configuration
        self.asset_sources = {
            'pexels': {
                'base_url': 'https://api.pexels.com/v1',
                'api_key': self.config.get('PEXELS_API_KEY', ''),
                'enabled': bool(self.config.get('PEXELS_API_KEY'))
            },
            'unsplash': {
                'base_url': 'https://api.unsplash.com',
                'api_key': self.config.get('UNSPLASH_API_KEY', ''),
                'enabled': bool(self.config.get('UNSPLASH_API_KEY'))
            },
            'pixabay': {
                'base_url': 'https://pixabay.com/api',
                'api_key': self.config.get('PIXABAY_API_KEY', ''),
                'enabled': bool(self.config.get('PIXABAY_API_KEY'))
            }
        }
        
        # Initialize templates
        self.templates = self._initialize_templates()
        
        # Performance tracking
        self.template_performance = {}
        self._load_template_performance()
        
        self.logger.info("Dynamic Video Template Manager initialized")
    
    def _initialize_templates(self) -> Dict[VideoTemplateType, VideoTemplate]:
        """Initialize available video templates"""
        templates = {}
        
        # List-Style Template
        templates[VideoTemplateType.LIST_STYLE] = VideoTemplate(
            template_type=VideoTemplateType.LIST_STYLE,
            name="List-Style Video",
            description="Countdown or list format with numbered items",
            structure=[
                {"type": "intro", "duration": 3, "content": "title_card"},
                {"type": "list_intro", "duration": 2, "content": "list_preview"},
                {"type": "list_items", "duration": 25, "content": "enumerated_items"},
                {"type": "conclusion", "duration": 5, "content": "summary_card"}
            ],
            asset_requirements={
                "background_images": 8,
                "transition_effects": 5,
                "background_music": 1,
                "sound_effects": 3
            },
            duration_range=(30, 60),
            style_config={
                "text_style": "bold_numbered",
                "transition_type": "slide_fade",
                "color_scheme": "vibrant",
                "music_mood": "upbeat"
            },
            compatibility_score=0.85
        )
        
        # Single Story Template
        templates[VideoTemplateType.SINGLE_STORY] = VideoTemplate(
            template_type=VideoTemplateType.SINGLE_STORY,
            name="Single Story Narrative",
            description="Focused narrative around one main story or event",
            structure=[
                {"type": "hook", "duration": 5, "content": "attention_grabber"},
                {"type": "setup", "duration": 8, "content": "story_context"},
                {"type": "main_content", "duration": 35, "content": "story_details"},
                {"type": "resolution", "duration": 7, "content": "story_conclusion"}
            ],
            asset_requirements={
                "background_images": 12,
                "background_videos": 5,
                "background_music": 1,
                "ambient_sounds": 2
            },
            duration_range=(45, 90),
            style_config={
                "text_style": "narrative",
                "transition_type": "cinematic",
                "color_scheme": "dramatic",
                "music_mood": "emotional"
            },
            compatibility_score=0.75
        )
        
        # Q&A Format Template
        templates[VideoTemplateType.QA_FORMAT] = VideoTemplate(
            template_type=VideoTemplateType.QA_FORMAT,
            name="Q&A Interactive Format",
            description="Question and answer format with audience engagement",
            structure=[
                {"type": "intro", "duration": 4, "content": "topic_introduction"},
                {"type": "question_setup", "duration": 3, "content": "question_preview"},
                {"type": "qa_segments", "duration": 28, "content": "question_answer_pairs"},
                {"type": "engagement", "duration": 5, "content": "call_to_action"}
            ],
            asset_requirements={
                "question_graphics": 6,
                "background_images": 10,
                "ui_elements": 8,
                "background_music": 1
            },
            duration_range=(35, 60),
            style_config={
                "text_style": "question_highlight",
                "transition_type": "quiz_style",
                "color_scheme": "engaging",
                "music_mood": "curious"
            },
            compatibility_score=0.80
        )
        
        # Tutorial Template
        templates[VideoTemplateType.TUTORIAL] = VideoTemplate(
            template_type=VideoTemplateType.TUTORIAL,
            name="Step-by-Step Tutorial",
            description="Educational content with clear step progression",
            structure=[
                {"type": "intro", "duration": 5, "content": "tutorial_overview"},
                {"type": "prerequisites", "duration": 3, "content": "requirements"},
                {"type": "steps", "duration": 40, "content": "tutorial_steps"},
                {"type": "summary", "duration": 7, "content": "recap_next_steps"}
            ],
            asset_requirements={
                "step_graphics": 8,
                "diagram_images": 5,
                "background_music": 1,
                "ui_elements": 10
            },
            duration_range=(45, 120),
            style_config={
                "text_style": "instructional",
                "transition_type": "step_progression",
                "color_scheme": "educational",
                "music_mood": "focused"
            },
            compatibility_score=0.70
        )
        
        # News Style Template
        templates[VideoTemplateType.NEWS_STYLE] = VideoTemplate(
            template_type=VideoTemplateType.NEWS_STYLE,
            name="News Bulletin Style",
            description="Breaking news or update format with urgency",
            structure=[
                {"type": "breaking", "duration": 3, "content": "breaking_news_intro"},
                {"type": "headline", "duration": 4, "content": "main_headline"},
                {"type": "details", "duration": 25, "content": "news_details"},
                {"type": "impact", "duration": 8, "content": "analysis_implications"}
            ],
            asset_requirements={
                "news_graphics": 6,
                "background_images": 8,
                "lower_thirds": 5,
                "news_music": 1
            },
            duration_range=(30, 60),
            style_config={
                "text_style": "news_ticker",
                "transition_type": "news_wipe",
                "color_scheme": "professional",
                "music_mood": "urgent"
            },
            compatibility_score=0.65
        )
        
        # Compilation Template
        templates[VideoTemplateType.COMPILATION] = VideoTemplate(
            template_type=VideoTemplateType.COMPILATION,
            name="Best-Of Compilation",
            description="Collection of highlights or best moments",
            structure=[
                {"type": "intro", "duration": 4, "content": "compilation_intro"},
                {"type": "montage", "duration": 35, "content": "highlight_montage"},
                {"type": "finale", "duration": 6, "content": "compilation_finale"}
            ],
            asset_requirements={
                "highlight_clips": 12,
                "transition_effects": 8,
                "background_music": 1,
                "sound_effects": 6
            },
            duration_range=(40, 75),
            style_config={
                "text_style": "highlight",
                "transition_type": "montage",
                "color_scheme": "energetic",
                "music_mood": "high_energy"
            },
            compatibility_score=0.90
        )
        
        return templates
    
    def _load_template_performance(self):
        """Load template performance data from file"""
        try:
            performance_file = Path("data/template_performance.json")
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    self.template_performance = json.load(f)
            else:
                # Initialize with default performance scores
                self.template_performance = {
                    template_type.value: {
                        'success_rate': 0.75,
                        'engagement_score': 70.0,
                        'usage_count': 0,
                        'last_updated': datetime.now().isoformat()
                    }
                    for template_type in VideoTemplateType
                }
        except Exception as e:
            self.logger.warning(f"Failed to load template performance: {e}")
            self.template_performance = {}
    
    def _save_template_performance(self):
        """Save template performance data to file"""
        try:
            performance_file = Path("data/template_performance.json")
            performance_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(performance_file, 'w') as f:
                json.dump(self.template_performance, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to save template performance: {e}")
    
    async def select_optimal_template(self, 
                                    content_analysis: Dict[str, Any],
                                    preferences: Optional[Dict[str, Any]] = None) -> VideoTemplate:
        """
        Select the optimal video template based on content analysis and preferences
        
        Args:
            content_analysis: Analysis results from AdvancedContentAnalyzer
            preferences: User preferences or constraints
            
        Returns:
            Selected VideoTemplate
        """
        try:
            self.logger.info("Selecting optimal video template")
            
            # Calculate compatibility scores for each template
            template_scores = {}
            
            for template_type, template in self.templates.items():
                score = await self._calculate_template_compatibility(
                    template, content_analysis, preferences
                )
                template_scores[template_type] = score
            
            # Select template with highest score
            best_template_type = max(template_scores, key=template_scores.get)
            selected_template = self.templates[best_template_type]
            
            self.logger.info(f"Selected template: {selected_template.name} (score: {template_scores[best_template_type]:.2f})")
            
            return selected_template
            
        except Exception as e:
            self.logger.error(f"Template selection failed: {e}")
            # Return default template
            return self.templates[VideoTemplateType.SINGLE_STORY]
    
    async def _calculate_template_compatibility(self,
                                              template: VideoTemplate,
                                              content_analysis: Dict[str, Any],
                                              preferences: Optional[Dict[str, Any]]) -> float:
        """Calculate compatibility score for a template"""
        try:
            score = template.compatibility_score * 100  # Base score
            
            # Content type compatibility
            content_keywords = content_analysis.get('keywords', [])
            content_topics = content_analysis.get('topics', [])
            
            # List-style works well with enumerable content
            if template.template_type == VideoTemplateType.LIST_STYLE:
                if any(keyword in ['top', 'best', 'worst', 'list', 'tips'] 
                       for keyword in content_keywords):
                    score += 20
                if 'numbers' in str(content_analysis.get('metadata', {})):
                    score += 15
            
            # Single story works well with narrative content
            elif template.template_type == VideoTemplateType.SINGLE_STORY:
                if any(topic in ['news', 'entertainment', 'lifestyle'] 
                       for topic in content_topics):
                    score += 25
                if content_analysis.get('engagement_potential', 0) > 70:
                    score += 15
            
            # Q&A works well with educational content
            elif template.template_type == VideoTemplateType.QA_FORMAT:
                if any(keyword in ['question', 'answer', 'how', 'why', 'what'] 
                       for keyword in content_keywords):
                    score += 25
                if 'education' in content_topics:
                    score += 20
            
            # Tutorial works well with how-to content
            elif template.template_type == VideoTemplateType.TUTORIAL:
                if any(keyword in ['tutorial', 'guide', 'how', 'learn', 'step'] 
                       for keyword in content_keywords):
                    score += 30
                if 'education' in content_topics:
                    score += 20
            
            # News style works well with current events
            elif template.template_type == VideoTemplateType.NEWS_STYLE:
                if any(keyword in ['news', 'breaking', 'update', 'report'] 
                       for keyword in content_keywords):
                    score += 25
                if 'news' in content_topics:
                    score += 20
            
            # Compilation works well with entertainment
            elif template.template_type == VideoTemplateType.COMPILATION:
                if any(topic in ['entertainment', 'gaming', 'sports'] 
                       for topic in content_topics):
                    score += 25
                if content_analysis.get('sentiment_score', 0) > 0.5:
                    score += 15
            
            # Performance history bonus
            template_perf = self.template_performance.get(template.template_type.value, {})
            success_rate = template_perf.get('success_rate', 0.75)
            engagement_score = template_perf.get('engagement_score', 70.0)
            
            score += (success_rate * 20)  # Up to 20 points for success rate
            score += (engagement_score - 50) * 0.3  # Engagement score influence
            
            # User preferences
            if preferences:
                preferred_duration = preferences.get('duration_preference', 45)
                min_dur, max_dur = template.duration_range
                
                if min_dur <= preferred_duration <= max_dur:
                    score += 10
                
                preferred_style = preferences.get('style_preference')
                if preferred_style and preferred_style in template.style_config.values():
                    score += 15
            
            return min(max(score, 0), 100)
            
        except Exception as e:
            self.logger.warning(f"Template compatibility calculation failed: {e}")
            return 50.0
    
    async def source_assets_for_template(self,
                                       template: VideoTemplate,
                                       content_keywords: List[str],
                                       content_topics: List[str]) -> Dict[str, List[VideoAsset]]:
        """
        Source assets automatically for the selected template
        
        Args:
            template: Selected video template
            content_keywords: Content keywords for asset search
            content_topics: Content topics for asset search
            
        Returns:
            Dictionary of asset types and their sourced assets
        """
        try:
            self.logger.info(f"Sourcing assets for template: {template.name}")
            
            sourced_assets = {}
            
            # Source different types of assets
            for asset_type, count in template.asset_requirements.items():
                assets = await self._source_specific_assets(
                    asset_type, count, content_keywords, content_topics, template.style_config
                )
                sourced_assets[asset_type] = assets
            
            total_assets = sum(len(assets) for assets in sourced_assets.values())
            self.logger.info(f"Sourced {total_assets} assets for template")
            
            return sourced_assets
            
        except Exception as e:
            self.logger.error(f"Asset sourcing failed: {e}")
            return {}
    
    async def _source_specific_assets(self,
                                    asset_type: str,
                                    count: int,
                                    keywords: List[str],
                                    topics: List[str],
                                    style_config: Dict[str, Any]) -> List[VideoAsset]:
        """Source specific type of assets"""
        try:
            assets = []
            
            # Determine search terms based on asset type and content
            search_terms = self._generate_search_terms(asset_type, keywords, topics)
            
            # Try different asset sources
            for source_name, source_config in self.asset_sources.items():
                if not source_config['enabled'] or len(assets) >= count:
                    continue
                
                source_assets = await self._fetch_from_source(
                    source_name, source_config, asset_type, search_terms, count - len(assets)
                )
                assets.extend(source_assets)
            
            # If not enough assets, generate fallback assets
            if len(assets) < count:
                fallback_assets = await self._generate_fallback_assets(
                    asset_type, count - len(assets), keywords, style_config
                )
                assets.extend(fallback_assets)
            
            return assets[:count]  # Return exactly the requested count
            
        except Exception as e:
            self.logger.warning(f"Specific asset sourcing failed for {asset_type}: {e}")
            return []
    
    def _generate_search_terms(self, asset_type: str, keywords: List[str], topics: List[str]) -> List[str]:
        """Generate search terms for asset sourcing"""
        search_terms = []
        
        # Base terms for different asset types
        asset_base_terms = {
            'background_images': ['background', 'texture', 'abstract', 'gradient'],
            'background_videos': ['motion', 'animation', 'abstract', 'background'],
            'transition_effects': ['transition', 'effect', 'animation'],
            'background_music': ['instrumental', 'background', 'ambient'],
            'sound_effects': ['sound', 'effect', 'audio'],
            'question_graphics': ['question', 'graphic', 'ui', 'interface'],
            'ui_elements': ['button', 'interface', 'graphic', 'design'],
            'news_graphics': ['news', 'broadcast', 'graphic', 'professional'],
            'lower_thirds': ['lower third', 'graphic', 'text', 'overlay']
        }
        
        # Add base terms
        base_terms = asset_base_terms.get(asset_type, ['general'])
        search_terms.extend(base_terms)
        
        # Add content-specific terms
        relevant_keywords = [kw for kw in keywords if len(kw) > 3][:3]
        search_terms.extend(relevant_keywords)
        
        # Add topic-specific terms
        search_terms.extend(topics[:2])
        
        return search_terms[:5]  # Limit to 5 search terms
    
    async def _fetch_from_source(self,
                               source_name: str,
                               source_config: Dict[str, Any],
                               asset_type: str,
                               search_terms: List[str],
                               count: int) -> List[VideoAsset]:
        """Fetch assets from a specific source"""
        try:
            if not REQUESTS_AVAILABLE:
                return []
            
            assets = []
            
            for search_term in search_terms:
                if len(assets) >= count:
                    break
                
                if source_name == 'pexels':
                    source_assets = await self._fetch_from_pexels(
                        source_config, asset_type, search_term, count - len(assets)
                    )
                elif source_name == 'unsplash':
                    source_assets = await self._fetch_from_unsplash(
                        source_config, asset_type, search_term, count - len(assets)
                    )
                elif source_name == 'pixabay':
                    source_assets = await self._fetch_from_pixabay(
                        source_config, asset_type, search_term, count - len(assets)
                    )
                else:
                    continue
                
                assets.extend(source_assets)
                
                # Small delay between requests
                await asyncio.sleep(0.5)
            
            return assets
            
        except Exception as e:
            self.logger.warning(f"Failed to fetch from {source_name}: {e}")
            return []
    
    async def _fetch_from_pexels(self,
                               config: Dict[str, Any],
                               asset_type: str,
                               search_term: str,
                               count: int) -> List[VideoAsset]:
        """Fetch assets from Pexels API"""
        try:
            assets = []
            
            # Determine Pexels endpoint based on asset type
            if 'video' in asset_type:
                endpoint = f"{config['base_url']}/videos/search"
            else:
                endpoint = f"{config['base_url']}/search"
            
            headers = {'Authorization': config['api_key']}
            params = {
                'query': search_term,
                'per_page': min(count, 15),
                'orientation': 'landscape'
            }
            
            response = requests.get(endpoint, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'video' in asset_type:
                    items = data.get('videos', [])
                    for item in items:
                        asset = VideoAsset(
                            url=item['video_files'][0]['link'] if item['video_files'] else '',
                            asset_type='video',
                            description=item.get('alt', search_term),
                            keywords=[search_term],
                            source='pexels',
                            license_type='free',
                            duration=item.get('duration'),
                            resolution=(item.get('width'), item.get('height'))
                        )
                        assets.append(asset)
                else:
                    items = data.get('photos', [])
                    for item in items:
                        asset = VideoAsset(
                            url=item['src']['large'],
                            asset_type='image',
                            description=item.get('alt', search_term),
                            keywords=[search_term],
                            source='pexels',
                            license_type='free',
                            resolution=(item.get('width'), item.get('height'))
                        )
                        assets.append(asset)
            
            return assets
            
        except Exception as e:
            self.logger.warning(f"Pexels API request failed: {e}")
            return []
    
    async def _fetch_from_unsplash(self,
                                 config: Dict[str, Any],
                                 asset_type: str,
                                 search_term: str,
                                 count: int) -> List[VideoAsset]:
        """Fetch assets from Unsplash API"""
        try:
            assets = []
            
            # Unsplash primarily provides images
            if 'video' in asset_type:
                return []
            
            endpoint = f"{config['base_url']}/search/photos"
            headers = {'Authorization': f"Client-ID {config['api_key']}"}
            params = {
                'query': search_term,
                'per_page': min(count, 20),
                'orientation': 'landscape'
            }
            
            response = requests.get(endpoint, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('results', [])
                
                for item in items:
                    asset = VideoAsset(
                        url=item['urls']['regular'],
                        asset_type='image',
                        description=item.get('alt_description', search_term),
                        keywords=[search_term],
                        source='unsplash',
                        license_type='free',
                        resolution=(item.get('width'), item.get('height'))
                    )
                    assets.append(asset)
            
            return assets
            
        except Exception as e:
            self.logger.warning(f"Unsplash API request failed: {e}")
            return []
    
    async def _fetch_from_pixabay(self,
                                config: Dict[str, Any],
                                asset_type: str,
                                search_term: str,
                                count: int) -> List[VideoAsset]:
        """Fetch assets from Pixabay API"""
        try:
            assets = []
            
            # Determine media type
            if 'video' in asset_type:
                media_type = 'video'
            else:
                media_type = 'photo'
            
            endpoint = config['base_url']
            params = {
                'key': config['api_key'],
                'q': search_term,
                'image_type': media_type,
                'per_page': min(count, 20),
                'safesearch': 'true'
            }
            
            response = requests.get(endpoint, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('hits', [])
                
                for item in items:
                    if media_type == 'video':
                        asset = VideoAsset(
                            url=item.get('videos', {}).get('medium', {}).get('url', ''),
                            asset_type='video',
                            description=item.get('tags', search_term),
                            keywords=[search_term],
                            source='pixabay',
                            license_type='free',
                            duration=item.get('duration'),
                            resolution=(item.get('imageWidth'), item.get('imageHeight'))
                        )
                    else:
                        asset = VideoAsset(
                            url=item['largeImageURL'],
                            asset_type='image',
                            description=item.get('tags', search_term),
                            keywords=[search_term],
                            source='pixabay',
                            license_type='free',
                            resolution=(item.get('imageWidth'), item.get('imageHeight'))
                        )
                    assets.append(asset)
            
            return assets
            
        except Exception as e:
            self.logger.warning(f"Pixabay API request failed: {e}")
            return []
    
    async def _generate_fallback_assets(self,
                                      asset_type: str,
                                      count: int,
                                      keywords: List[str],
                                      style_config: Dict[str, Any]) -> List[VideoAsset]:
        """Generate fallback assets when external sources fail"""
        try:
            assets = []
            
            # Generate simple placeholder assets
            for i in range(count):
                if 'image' in asset_type or 'graphic' in asset_type:
                    asset = await self._generate_placeholder_image(
                        asset_type, keywords, style_config, i
                    )
                elif 'video' in asset_type:
                    asset = await self._generate_placeholder_video(
                        asset_type, keywords, style_config, i
                    )
                else:
                    # For audio and effects, create placeholder entries
                    asset = VideoAsset(
                        url=f"placeholder_{asset_type}_{i}.mp3",
                        asset_type='audio',
                        description=f"Generated {asset_type} {i+1}",
                        keywords=keywords,
                        source='generated',
                        license_type='free'
                    )
                
                if asset:
                    assets.append(asset)
            
            return assets
            
        except Exception as e:
            self.logger.warning(f"Fallback asset generation failed: {e}")
            return []
    
    async def _generate_placeholder_image(self,
                                        asset_type: str,
                                        keywords: List[str],
                                        style_config: Dict[str, Any],
                                        index: int) -> Optional[VideoAsset]:
        """Generate a placeholder image"""
        try:
            if not PIL_AVAILABLE:
                return None
            
            # Create a simple gradient or solid color image
            width, height = 1920, 1080
            
            # Color scheme based on style config
            color_scheme = style_config.get('color_scheme', 'vibrant')
            
            colors = {
                'vibrant': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
                'dramatic': ['#2C3E50', '#34495E', '#7F8C8D', '#95A5A6', '#BDC3C7'],
                'engaging': ['#E74C3C', '#3498DB', '#9B59B6', '#E67E22', '#F39C12'],
                'educational': ['#2ECC71', '#3498DB', '#9B59B6', '#E67E22', '#F39C12'],
                'professional': ['#34495E', '#2C3E50', '#7F8C8D', '#95A5A6', '#BDC3C7'],
                'energetic': ['#FF5733', '#C70039', '#900C3F', '#581845', '#FFC300']
            }
            
            scheme_colors = colors.get(color_scheme, colors['vibrant'])
            color = scheme_colors[index % len(scheme_colors)]
            
            # Create image
            img = Image.new('RGB', (width, height), color)
            
            # Add some text if it's a graphic type
            if 'graphic' in asset_type:
                draw = ImageDraw.Draw(img)
                text = f"{asset_type.replace('_', ' ').title()} {index + 1}"
                
                # Try to use a font, fallback to default
                try:
                    font = ImageFont.truetype("arial.ttf", 48)
                except:
                    font = ImageFont.load_default()
                
                # Center text
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                position = ((width - text_width) // 2, (height - text_height) // 2)
                
                draw.text(position, text, fill='white', font=font)
            
            # Save to temporary location
            temp_path = Path(f"data/temp/generated_{asset_type}_{index}.png")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(temp_path)
            
            return VideoAsset(
                url=str(temp_path),
                asset_type='image',
                description=f"Generated {asset_type} {index + 1}",
                keywords=keywords,
                source='generated',
                license_type='free',
                resolution=(width, height)
            )
            
        except Exception as e:
            self.logger.warning(f"Placeholder image generation failed: {e}")
            return None
    
    async def _generate_placeholder_video(self,
                                        asset_type: str,
                                        keywords: List[str],
                                        style_config: Dict[str, Any],
                                        index: int) -> Optional[VideoAsset]:
        """Generate a placeholder video (simplified)"""
        try:
            # For now, return a placeholder entry
            # In a full implementation, this could generate simple animated videos
            return VideoAsset(
                url=f"placeholder_{asset_type}_{index}.mp4",
                asset_type='video',
                description=f"Generated {asset_type} {index + 1}",
                keywords=keywords,
                source='generated',
                license_type='free',
                duration=5.0,
                resolution=(1920, 1080)
            )
            
        except Exception as e:
            self.logger.warning(f"Placeholder video generation failed: {e}")
            return None
    
    async def update_template_performance(self,
                                        template_type: VideoTemplateType,
                                        success: bool,
                                        engagement_score: Optional[float] = None):
        """Update template performance metrics"""
        try:
            template_key = template_type.value
            
            if template_key not in self.template_performance:
                self.template_performance[template_key] = {
                    'success_rate': 0.75,
                    'engagement_score': 70.0,
                    'usage_count': 0,
                    'last_updated': datetime.now().isoformat()
                }
            
            perf = self.template_performance[template_key]
            
            # Update success rate (weighted average)
            current_rate = perf['success_rate']
            weight = min(perf['usage_count'], 10) / 10.0  # Weight based on usage count
            new_rate = (current_rate * weight + (1.0 if success else 0.0) * (1 - weight))
            perf['success_rate'] = new_rate
            
            # Update engagement score if provided
            if engagement_score is not None:
                current_engagement = perf['engagement_score']
                new_engagement = (current_engagement * weight + engagement_score * (1 - weight))
                perf['engagement_score'] = new_engagement
            
            # Update usage count and timestamp
            perf['usage_count'] += 1
            perf['last_updated'] = datetime.now().isoformat()
            
            # Save performance data
            self._save_template_performance()
            
            self.logger.info(f"Updated performance for {template_type.value}: "
                           f"success_rate={perf['success_rate']:.3f}, "
                           f"engagement={perf['engagement_score']:.1f}")
            
        except Exception as e:
            self.logger.warning(f"Failed to update template performance: {e}")
    
    def get_template_analytics(self) -> Dict[str, Any]:
        """Get template analytics and performance data"""
        return {
            'total_templates': len(self.templates),
            'performance_data': self.template_performance,
            'asset_sources': {
                name: {'enabled': config['enabled'], 'available': bool(config['api_key'])}
                for name, config in self.asset_sources.items()
            },
            'last_updated': datetime.now().isoformat()
        }