"""
Enhanced Thumbnail Generator with A/B Testing and AI-Driven Optimization
Creates multiple thumbnail variants and tracks their performance for optimization.
"""

import logging
import tempfile
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter, ImageOps
from moviepy import VideoFileClip
from datetime import datetime, timedelta

from src.config.settings import get_config
from src.models import VideoAnalysisEnhanced, ThumbnailVariant, PerformanceMetrics
from src.processing.thumbnail_generator import ThumbnailGenerator
from src.monitoring.engagement_metrics import EngagementMetricsDB


class EnhancedThumbnailGenerator(ThumbnailGenerator):
    """
    Enhanced thumbnail generator with A/B testing, performance tracking,
    and AI-driven optimization capabilities.
    """
    
    def __init__(self):
        super().__init__()
        self.engagement_db = EngagementMetricsDB()
        
        # A/B testing parameters
        self.max_variants = 5
        self.performance_window_days = 7
        
        # AI optimization parameters
        self.color_schemes = {
            'high_contrast': {'text_color': 'white', 'stroke_color': 'black', 'bg_overlay': 0.3},
            'warm_bright': {'text_color': 'yellow', 'stroke_color': 'red', 'bg_overlay': 0.2},
            'cool_dramatic': {'text_color': 'cyan', 'stroke_color': 'blue', 'bg_overlay': 0.4},
            'classic_bold': {'text_color': 'white', 'stroke_color': 'black', 'bg_overlay': 0.1},
            'neon_pop': {'text_color': 'lime', 'stroke_color': 'magenta', 'bg_overlay': 0.25}
        }
        
        self.text_styles = {
            'bold_impact': {'font_weight': 'bold', 'size_factor': 1.2, 'spacing': 1.1},
            'dramatic_tall': {'font_weight': 'bold', 'size_factor': 1.0, 'spacing': 1.3},
            'compact_punch': {'font_weight': 'bold', 'size_factor': 0.9, 'spacing': 0.9},
            'elegant_modern': {'font_weight': 'normal', 'size_factor': 1.1, 'spacing': 1.0},
            'explosive_caps': {'font_weight': 'bold', 'size_factor': 1.3, 'spacing': 1.2}
        }
        
        self.emotional_tones = {
            'exciting': {'saturation': 1.3, 'contrast': 1.2, 'brightness': 1.1},
            'dramatic': {'saturation': 0.9, 'contrast': 1.4, 'brightness': 0.9},
            'mysterious': {'saturation': 0.7, 'contrast': 1.3, 'brightness': 0.8},
            'energetic': {'saturation': 1.4, 'contrast': 1.1, 'brightness': 1.2},
            'professional': {'saturation': 1.0, 'contrast': 1.1, 'brightness': 1.0}
        }
    
    def generate_ab_test_thumbnails(self, 
                                   video_path: Path,
                                   analysis: VideoAnalysisEnhanced,
                                   output_dir: Path,
                                   num_variants: int = 3) -> List[ThumbnailVariant]:
        """
        Generate multiple thumbnail variants for A/B testing
        
        Args:
            video_path: Source video path
            analysis: Enhanced video analysis
            output_dir: Directory to save variants
            num_variants: Number of variants to generate
            
        Returns:
            List of thumbnail variants with metadata
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            variants = []
            
            # Get performance data for optimization
            performance_data = self._get_historical_performance()
            
            # Generate base variant configurations
            variant_configs = self._generate_variant_configs(analysis, num_variants, performance_data)
            
            for i, config in enumerate(variant_configs):
                try:
                    variant_id = f"variant_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    variant_path = output_dir / f"{variant_id}.jpg"
                    
                    # Generate thumbnail with specific configuration
                    success = self._generate_optimized_thumbnail(
                        video_path, analysis, variant_path, config
                    )
                    
                    if success:
                        variant = ThumbnailVariant(
                            variant_id=variant_id,
                            headline_text=config['headline_text'],
                            timestamp_seconds=config['timestamp_seconds'],
                            text_style=config['text_style'],
                            color_scheme=config['color_scheme'],
                            emotional_tone=config['emotional_tone']
                        )
                        variants.append(variant)
                        
                        # Store variant configuration for tracking
                        self._store_variant_config(variant_id, config, str(variant_path))
                        
                        self.logger.info(f"Generated A/B test variant: {variant_id}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to generate variant {i+1}: {e}")
            
            self.logger.info(f"Generated {len(variants)} A/B test thumbnail variants")
            return variants
            
        except Exception as e:
            self.logger.error(f"A/B test thumbnail generation failed: {e}")
            return []
    
    def _get_historical_performance(self) -> Dict[str, Any]:
        """Get historical thumbnail performance data for optimization"""
        try:
            # Query recent performance data
            cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()
            
            # This would integrate with YouTube Analytics API in production
            # For now, we'll use placeholder data structure
            performance_data = {
                'avg_ctr_by_style': {
                    'bold_impact': 0.15,
                    'dramatic_tall': 0.12,
                    'compact_punch': 0.18,
                    'elegant_modern': 0.10,
                    'explosive_caps': 0.16
                },
                'avg_ctr_by_color': {
                    'high_contrast': 0.14,
                    'warm_bright': 0.16,
                    'cool_dramatic': 0.11,
                    'classic_bold': 0.13,
                    'neon_pop': 0.17
                },
                'avg_ctr_by_tone': {
                    'exciting': 0.15,
                    'dramatic': 0.13,
                    'mysterious': 0.11,
                    'energetic': 0.17,
                    'professional': 0.09
                },
                'best_performing_combinations': [
                    {'style': 'compact_punch', 'color': 'neon_pop', 'tone': 'energetic', 'ctr': 0.21},
                    {'style': 'explosive_caps', 'color': 'warm_bright', 'tone': 'exciting', 'ctr': 0.19},
                    {'style': 'bold_impact', 'color': 'high_contrast', 'tone': 'dramatic', 'ctr': 0.17}
                ]
            }
            
            return performance_data
            
        except Exception as e:
            self.logger.warning(f"Could not retrieve performance data: {e}")
            return {}
    
    def _generate_variant_configs(self, 
                                 analysis: VideoAnalysisEnhanced,
                                 num_variants: int,
                                 performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate configurations for thumbnail variants"""
        configs = []
        
        # Get available headlines
        headlines = [analysis.hook_text] + analysis.hook_variations[:4]
        
        # Get optimal timestamps for different variants
        timestamps = self._calculate_optimal_timestamps(analysis, num_variants)
        
        # Generate configurations using AI optimization
        for i in range(num_variants):
            config = self._generate_optimized_config(
                i, headlines, timestamps, performance_data, analysis
            )
            configs.append(config)
        
        return configs
    
    def _generate_optimized_config(self, 
                                  variant_index: int,
                                  headlines: List[str],
                                  timestamps: List[float],
                                  performance_data: Dict[str, Any],
                                  analysis: VideoAnalysisEnhanced) -> Dict[str, Any]:
        """Generate an optimized configuration for a variant"""
        
        # Strategy selection based on variant index
        if variant_index == 0:
            # Control variant - use best performing combination
            if performance_data.get('best_performing_combinations'):
                best = performance_data['best_performing_combinations'][0]
                return {
                    'headline_text': headlines[0] if headlines else "Must Watch!",
                    'timestamp_seconds': timestamps[0] if timestamps else 10.0,
                    'text_style': best['style'],
                    'color_scheme': best['color'],
                    'emotional_tone': best['tone'],
                    'optimization_strategy': 'best_known'
                }
        
        # Experimental variants
        strategies = ['high_contrast', 'emotional_impact', 'curiosity_gap', 'trend_following']
        strategy = strategies[variant_index % len(strategies)]
        
        if strategy == 'high_contrast':
            return {
                'headline_text': headlines[variant_index % len(headlines)] if headlines else "Amazing!",
                'timestamp_seconds': timestamps[variant_index % len(timestamps)] if timestamps else 5.0,
                'text_style': 'bold_impact',
                'color_scheme': 'high_contrast',
                'emotional_tone': 'dramatic',
                'optimization_strategy': 'high_contrast'
            }
        
        elif strategy == 'emotional_impact':
            return {
                'headline_text': headlines[variant_index % len(headlines)] if headlines else "Incredible!",
                'timestamp_seconds': timestamps[variant_index % len(timestamps)] if timestamps else 15.0,
                'text_style': 'explosive_caps',
                'color_scheme': 'warm_bright',
                'emotional_tone': 'exciting',
                'optimization_strategy': 'emotional_impact'
            }
        
        elif strategy == 'curiosity_gap':
            # Use more mysterious/intriguing elements
            curiosity_headlines = [h for h in headlines if any(word in h.lower() 
                                 for word in ['secret', 'hidden', 'revealed', 'truth', 'mystery'])]
            headline = curiosity_headlines[0] if curiosity_headlines else (headlines[0] if headlines else "You Won't Believe...")
            
            return {
                'headline_text': headline,
                'timestamp_seconds': timestamps[variant_index % len(timestamps)] if timestamps else 20.0,
                'text_style': 'dramatic_tall',
                'color_scheme': 'cool_dramatic',
                'emotional_tone': 'mysterious',
                'optimization_strategy': 'curiosity_gap'
            }
        
        else:  # trend_following
            return {
                'headline_text': headlines[variant_index % len(headlines)] if headlines else "Viral!",
                'timestamp_seconds': timestamps[variant_index % len(timestamps)] if timestamps else 8.0,
                'text_style': 'compact_punch',
                'color_scheme': 'neon_pop',
                'emotional_tone': 'energetic',
                'optimization_strategy': 'trend_following'
            }
    
    def _calculate_optimal_timestamps(self, 
                                    analysis: VideoAnalysisEnhanced,
                                    num_variants: int) -> List[float]:
        """Calculate optimal timestamps for thumbnail variants"""
        timestamps = []
        
        # Use AI-suggested moments first
        if analysis.key_focus_points:
            focus_times = [fp.timestamp_seconds for fp in analysis.key_focus_points]
            timestamps.extend(focus_times[:num_variants])
        
        # Add dramatic moments
        if hasattr(analysis, 'visual_hook_moment') and analysis.visual_hook_moment:
            timestamps.append(analysis.visual_hook_moment.timestamp_seconds)
        
        # Add segment-based timestamps
        if analysis.segments:
            for segment in analysis.segments[:num_variants]:
                mid_point = (segment.start_seconds + segment.end_seconds) / 2
                timestamps.append(mid_point)
        
        # Fill with evenly distributed timestamps if needed
        while len(timestamps) < num_variants:
            position = len(timestamps) / (num_variants - 1) if num_variants > 1 else 0.5
            estimated_duration = analysis.original_duration or 60.0
            timestamp = position * estimated_duration * 0.8 + estimated_duration * 0.1  # 10% to 90%
            timestamps.append(timestamp)
        
        return timestamps[:num_variants]
    
    def _generate_optimized_thumbnail(self, 
                                    video_path: Path,
                                    analysis: VideoAnalysisEnhanced,
                                    output_path: Path,
                                    config: Dict[str, Any]) -> bool:
        """Generate thumbnail with specific optimization configuration"""
        try:
            # Extract frame at specified timestamp
            frame = self._extract_optimal_frame_with_config(video_path, config)
            if frame is None:
                return False
            
            # Convert to PIL Image
            pil_image = self._cv2_to_pil(frame)
            
            # Apply emotional tone adjustments
            enhanced_image = self._apply_emotional_tone(pil_image, config['emotional_tone'])
            
            # Add optimized text overlay
            text_image = self._add_optimized_text(enhanced_image, config)
            
            # Apply final enhancements
            final_image = self._apply_variant_enhancements(text_image, config)
            
            # Save thumbnail
            final_image.save(str(output_path), 'JPEG', quality=95, optimize=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Optimized thumbnail generation failed: {e}")
            return False
    
    def _extract_optimal_frame_with_config(self, video_path: Path, 
                                         config: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract frame with configuration-specific optimization"""
        try:
            timestamp = config['timestamp_seconds']
            
            with VideoFileClip(str(video_path)) as clip:
                # Ensure timestamp is within bounds
                timestamp = min(timestamp, clip.duration - 0.1)
                timestamp = max(0, timestamp)
                
                # For high-contrast strategy, find frame with good contrast
                if config.get('optimization_strategy') == 'high_contrast':
                    timestamp = self._find_high_contrast_frame(clip, timestamp)
                
                # Extract frame
                frame = clip.get_frame(timestamp)
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
        except Exception as e:
            self.logger.warning(f"Frame extraction failed: {e}")
            return None
    
    def _find_high_contrast_frame(self, clip: VideoFileClip, 
                                 target_time: float, search_window: float = 2.0) -> float:
        """Find frame with highest contrast around target time"""
        try:
            start_time = max(0, target_time - search_window/2)
            end_time = min(clip.duration, target_time + search_window/2)
            
            best_time = target_time
            best_contrast = 0
            
            # Sample frames in the window
            for t in np.linspace(start_time, end_time, 10):
                frame = clip.get_frame(t)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                contrast = gray.std()
                
                if contrast > best_contrast:
                    best_contrast = contrast
                    best_time = t
            
            return best_time
            
        except Exception as e:
            self.logger.debug(f"High contrast frame search failed: {e}")
            return target_time
    
    def _apply_emotional_tone(self, image: Image.Image, tone: str) -> Image.Image:
        """Apply emotional tone adjustments to image"""
        try:
            tone_config = self.emotional_tones.get(tone, self.emotional_tones['exciting'])
            
            # Apply saturation adjustment
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(tone_config['saturation'])
            
            # Apply contrast adjustment
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(tone_config['contrast'])
            
            # Apply brightness adjustment
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(tone_config['brightness'])
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Emotional tone application failed: {e}")
            return image
    
    def _add_optimized_text(self, image: Image.Image, config: Dict[str, Any]) -> Image.Image:
        """Add optimized text overlay based on configuration"""
        try:
            text = config['headline_text']
            style_config = self.text_styles.get(config['text_style'], self.text_styles['bold_impact'])
            color_config = self.color_schemes.get(config['color_scheme'], self.color_schemes['high_contrast'])
            
            # Create working copy
            img_with_text = image.copy()
            draw = ImageDraw.Draw(img_with_text)
            
            # Load font with style-specific sizing
            font_path = self.config.get_font_path('BebasNeue-Regular.ttf')
            base_font_size = min(120, image.width // (len(text) // 2 + 5))
            font_size = int(base_font_size * style_config['size_factor'])
            font_size = max(40, font_size)
            
            try:
                font = ImageFont.truetype(str(font_path), font_size)
            except Exception:
                font = ImageFont.load_default()
            
            # Calculate text positioning with style-specific spacing
            bbox = draw.textbbox((0, 0), text, font=font, spacing=font_size * 0.1)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (image.width - text_width) // 2
            y = int(image.height * 0.15)  # Upper portion
            
            # Add background overlay if specified
            if color_config['bg_overlay'] > 0:
                overlay = Image.new('RGBA', image.size, (0, 0, 0, int(255 * color_config['bg_overlay'])))
                img_with_text = Image.alpha_composite(img_with_text.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(img_with_text)
            
            # Draw text with style-specific stroke
            stroke_width = max(3, font_size // 15)
            
            # Draw stroke
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, font=font, fill=color_config['stroke_color'])
            
            # Draw main text
            draw.text((x, y), text, font=font, fill=color_config['text_color'])
            
            return img_with_text
            
        except Exception as e:
            self.logger.warning(f"Optimized text overlay failed: {e}")
            return image
    
    def _apply_variant_enhancements(self, image: Image.Image, config: Dict[str, Any]) -> Image.Image:
        """Apply variant-specific final enhancements"""
        try:
            strategy = config.get('optimization_strategy', 'default')
            
            if strategy == 'high_contrast':
                # Additional contrast boost
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.1)
            
            elif strategy == 'emotional_impact':
                # Slight warmth and saturation boost
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1.15)
            
            elif strategy == 'curiosity_gap':
                # Slight desaturation for mystery
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(0.9)
                # Add subtle vignette
                image = self._add_subtle_vignette(image)
            
            elif strategy == 'trend_following':
                # Bright and punchy
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1.05)
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Variant enhancement failed: {e}")
            return image
    
    def _add_subtle_vignette(self, image: Image.Image) -> Image.Image:
        """Add subtle vignette effect"""
        try:
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            # Create vignette mask
            y, x = np.ogrid[:height, :width]
            center_y, center_x = height // 2, width // 2
            
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            
            # Subtle vignette (less aggressive than main version)
            vignette = 1 - (distance / max_distance) * 0.15
            vignette = np.clip(vignette, 0.85, 1.0)
            
            # Apply vignette
            img_array = img_array * vignette[..., np.newaxis]
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
            return Image.fromarray(img_array)
            
        except Exception as e:
            self.logger.debug(f"Vignette application failed: {e}")
            return image
    
    def _store_variant_config(self, variant_id: str, config: Dict[str, Any], file_path: str):
        """Store variant configuration for performance tracking"""
        try:
            config_data = {
                'variant_id': variant_id,
                'config': config,
                'file_path': file_path,
                'created_at': datetime.now().isoformat(),
                'performance_data': None
            }
            
            # Store in database or file system
            config_path = Path(file_path).parent / f"{variant_id}_config.json"
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to store variant config: {e}")
    
    def update_variant_performance(self, variant_id: str, performance_metrics: PerformanceMetrics):
        """Update performance metrics for a thumbnail variant"""
        try:
            # Calculate performance score
            performance_score = self._calculate_performance_score(performance_metrics)
            
            # Store performance data
            self.engagement_db.store_video_metrics(performance_metrics)
            
            # Update variant with performance data
            # This would integrate with the variant tracking system
            self.logger.info(f"Updated performance for variant {variant_id}: score {performance_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"Failed to update variant performance: {e}")
    
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall performance score for a thumbnail"""
        try:
            # Weighted scoring system
            weights = {
                'ctr': 0.4,           # Click-through rate is most important
                'retention': 0.3,      # Viewer retention
                'engagement': 0.2,     # Likes, comments, shares
                'completion': 0.1      # Video completion rate
            }
            
            # Normalize metrics to 0-100 scale
            ctr_score = min(metrics.click_through_rate * 1000, 100)  # CTR is typically small
            retention_score = metrics.watch_time_percentage
            
            engagement_rate = 0
            if metrics.views > 0:
                total_engagement = metrics.likes + metrics.comments + metrics.shares
                engagement_rate = (total_engagement / metrics.views) * 100
                engagement_rate = min(engagement_rate, 100)
            
            completion_score = metrics.watch_time_percentage  # Proxy for completion
            
            # Calculate weighted score
            total_score = (
                ctr_score * weights['ctr'] +
                retention_score * weights['retention'] +
                engagement_rate * weights['engagement'] +
                completion_score * weights['completion']
            )
            
            return min(total_score, 100)
            
        except Exception as e:
            self.logger.warning(f"Performance score calculation failed: {e}")
            return 50.0  # Default average score
    
    def get_best_performing_variant(self, variants: List[ThumbnailVariant]) -> Optional[ThumbnailVariant]:
        """Identify the best performing thumbnail variant"""
        try:
            if not variants:
                return None
            
            best_variant = None
            best_score = 0
            
            for variant in variants:
                if variant.performance_score and variant.performance_score > best_score:
                    best_score = variant.performance_score
                    best_variant = variant
            
            return best_variant
            
        except Exception as e:
            self.logger.error(f"Failed to identify best variant: {e}")
            return variants[0] if variants else None