"""
Enhanced thumbnail generation with AI-powered text overlays and branding.
Creates compelling, clickable thumbnails that drive engagement.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from moviepy import VideoFileClip
from datetime import datetime, timedelta

from src.config.settings import get_config
from src.models import (
    VideoAnalysis,
    VideoAnalysisEnhanced,
    ThumbnailVariant,
    PerformanceMetrics,
)
from src.monitoring.engagement_metrics import EngagementMetricsDB


class ThumbnailGenerator:
    """
    Advanced thumbnail generator with AI-powered text overlays,
    visual enhancements, and branding integration.
    """

    def __init__(self, enable_ab_testing: bool = False):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.enable_ab_testing = enable_ab_testing
        self.engagement_db = EngagementMetricsDB()

        # A/B testing parameters
        self.max_variants = 5
        self.performance_window_days = 7

        # AI optimization parameters
        self.color_schemes = {
            "high_contrast": {
                "text_color": "white",
                "stroke_color": "black",
                "bg_overlay": 0.3,
            },
            "warm_bright": {
                "text_color": "yellow",
                "stroke_color": "red",
                "bg_overlay": 0.2,
            },
            "cool_dramatic": {
                "text_color": "cyan",
                "stroke_color": "blue",
                "bg_overlay": 0.4,
            },
            "classic_bold": {
                "text_color": "white",
                "stroke_color": "black",
                "bg_overlay": 0.1,
            },
            "neon_pop": {
                "text_color": "lime",
                "stroke_color": "magenta",
                "bg_overlay": 0.25,
            },
        }

        self.text_styles = {
            "bold_impact": {"font_weight": "bold", "size_factor": 1.2, "spacing": 1.1},
            "dramatic_tall": {
                "font_weight": "bold",
                "size_factor": 1.0,
                "spacing": 1.3,
            },
            "compact_punch": {
                "font_weight": "bold",
                "size_factor": 0.9,
                "spacing": 0.9,
            },
            "elegant_modern": {
                "font_weight": "normal",
                "size_factor": 1.1,
                "spacing": 1.0,
            },
            "explosive_caps": {
                "font_weight": "bold",
                "size_factor": 1.3,
                "spacing": 1.2,
            },
        }

        self.emotional_tones = {
            "exciting": {"saturation": 1.3, "contrast": 1.2, "brightness": 1.1},
            "dramatic": {"saturation": 0.9, "contrast": 1.4, "brightness": 0.9},
            "mysterious": {"saturation": 0.7, "contrast": 1.3, "brightness": 0.8},
            "energetic": {"saturation": 1.4, "contrast": 1.1, "brightness": 1.2},
            "professional": {"saturation": 1.0, "contrast": 1.1, "brightness": 1.0},
        }

    def generate_thumbnail(
        self, video_path: Path, analysis: VideoAnalysis, output_path: Path
    ) -> bool:
        """
        Generate an engaging thumbnail with AI-suggested elements

        Args:
            video_path: Path to source video
            analysis: AI analysis with thumbnail suggestions
            output_path: Path to save thumbnail

        Returns:
            True if thumbnail generated successfully
        """
        try:
            # Extract optimal frame
            frame = self._extract_optimal_frame(video_path, analysis)
            if frame is None:
                self.logger.error("Failed to extract frame for thumbnail")
                return False

            # Convert to PIL Image for processing
            pil_image = self._cv2_to_pil(frame)

            # Apply visual enhancements
            enhanced_image = self._enhance_image(pil_image)

            # Add headline text if provided
            if analysis.thumbnail_info.headline_text:
                enhanced_image = self._add_headline_text(
                    enhanced_image, analysis.thumbnail_info.headline_text
                )

            # Add branding/watermark if configured
            if hasattr(self.config, "watermark_path") and self.config.watermark_path:
                enhanced_image = self._add_watermark(enhanced_image)

            # Apply final polish
            final_image = self._apply_final_polish(enhanced_image)

            # Save thumbnail
            final_image.save(str(output_path), "JPEG", quality=95, optimize=True)

            self.logger.info(f"Generated thumbnail: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error generating thumbnail: {e}")
            return False

    def _extract_optimal_frame(
        self, video_path: Path, analysis: VideoAnalysis
    ) -> Optional[np.ndarray]:
        """Extract the best frame for thumbnail based on AI analysis"""
        try:
            timestamp = analysis.thumbnail_info.timestamp_seconds

            with VideoFileClip(str(video_path)) as clip:
                # Ensure timestamp is within video duration
                timestamp = min(timestamp, clip.duration - 0.1)
                timestamp = max(0, timestamp)

                # Extract frame
                frame = clip.get_frame(timestamp)

                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                return frame_bgr

        except Exception as e:
            self.logger.warning(f"Error extracting frame at {timestamp}s: {e}")
            # Fallback to middle frame
            try:
                with VideoFileClip(str(video_path)) as clip:
                    mid_timestamp = clip.duration / 2
                    frame = clip.get_frame(mid_timestamp)
                    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception as e2:
                self.logger.error(f"Fallback frame extraction failed: {e2}")
                return None

    def _cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL Image"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)

    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply visual enhancements to make thumbnail more appealing"""
        try:
            # Resize to YouTube thumbnail dimensions (1280x720)
            target_size = (1280, 720)

            # Calculate crop/resize to maintain aspect ratio
            original_ratio = image.width / image.height
            target_ratio = target_size[0] / target_size[1]

            if original_ratio > target_ratio:
                # Image is wider, crop width
                new_height = image.height
                new_width = int(new_height * target_ratio)
                left = (image.width - new_width) // 2
                image = image.crop((left, 0, left + new_width, new_height))
            else:
                # Image is taller, crop height
                new_width = image.width
                new_height = int(new_width / target_ratio)
                top = (image.height - new_height) // 2
                image = image.crop((0, top, new_width, top + new_height))

            # Resize to target dimensions
            image = image.resize(target_size, Image.Resampling.LANCZOS)

            # Enhance contrast and saturation
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)  # 20% more contrast

            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.1)  # 10% more saturation

            # Slight sharpening
            image = image.filter(
                ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3)
            )

            return image

        except Exception as e:
            self.logger.warning(f"Image enhancement failed: {e}")
            return image

    def _add_headline_text(self, image: Image.Image, headline: str) -> Image.Image:
        """Add compelling headline text to thumbnail"""
        try:
            # Create a copy to work with
            img_with_text = image.copy()
            draw = ImageDraw.Draw(img_with_text)

            # Load font
            font_path = self.config.get_font_path("BebasNeue-Regular.ttf")

            # Calculate font size based on image width and text length
            base_font_size = min(100, image.width // (len(headline) // 2 + 5))
            font_size = max(40, base_font_size)  # Minimum 40px

            try:
                font = ImageFont.truetype(str(font_path), font_size)
            except Exception:
                # Fallback to default font
                font = ImageFont.load_default()

            # Calculate text position (upper portion of image)
            bbox = draw.textbbox((0, 0), headline, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Position text in upper portion with padding
            x = (image.width - text_width) // 2
            y = image.height // 6  # Upper third area

            # Add text stroke/outline for visibility
            stroke_width = max(2, font_size // 20)

            # Draw text with stroke
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), headline, font=font, fill="black")

            # Draw main text
            draw.text((x, y), headline, font=font, fill="white")

            # Add subtle shadow
            shadow_offset = max(2, font_size // 25)
            draw.text(
                (x + shadow_offset, y + shadow_offset), headline, font=font, fill="gray"
            )
            draw.text(
                (x, y), headline, font=font, fill="white"
            )  # Redraw main text on top

            self.logger.debug(f"Added headline text: '{headline}'")
            return img_with_text

        except Exception as e:
            self.logger.warning(f"Failed to add headline text: {e}")
            return image

    def _add_watermark(self, image: Image.Image) -> Image.Image:
        """Add watermark/branding to thumbnail"""
        try:
            watermark_path = Path(self.config.watermark_path)
            if not watermark_path.exists():
                return image

            # Load watermark
            with Image.open(watermark_path) as img:
                watermark = img.convert("RGBA")

                # Resize watermark to be proportional to image
                max_watermark_size = min(image.width // 6, image.height // 6)
                watermark.thumbnail(
                    (max_watermark_size, max_watermark_size), Image.Resampling.LANCZOS
                )

                # Position watermark in bottom right corner with padding
                padding = 20
                position = (
                    image.width - watermark.width - padding,
                    image.height - watermark.height - padding,
                )

                # Create a copy and paste watermark
                img_with_watermark = image.copy().convert("RGBA")
                img_with_watermark.paste(watermark, position, watermark)

                # Convert back to RGB
                final_image = Image.new("RGB", img_with_watermark.size, (255, 255, 255))
                final_image.paste(
                    img_with_watermark, mask=img_with_watermark.split()[-1]
                )

                self.logger.debug("Added watermark to thumbnail")
                return final_image

        except Exception as e:
            self.logger.warning(f"Failed to add watermark: {e}")
            return image

    def _apply_final_polish(self, image: Image.Image) -> Image.Image:
        """Apply final polish and adjustments"""
        try:
            # Slight vignette effect for focus
            img_array = np.array(image)

            # Create vignette mask
            height, width = img_array.shape[:2]
            y, x = np.ogrid[:height, :width]
            center_y, center_x = height // 2, width // 2

            # Calculate distance from center
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            max_distance = np.sqrt(center_x**2 + center_y**2)

            # Create vignette (darker edges)
            vignette = 1 - (distance / max_distance) * 0.3  # 30% maximum darkening
            vignette = np.clip(vignette, 0.7, 1.0)  # Limit darkening

            # Apply vignette
            img_array = img_array * vignette[..., np.newaxis]
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

            polished_image = Image.fromarray(img_array)

            # Final brightness adjustment
            enhancer = ImageEnhance.Brightness(polished_image)
            polished_image = enhancer.enhance(1.05)  # Slightly brighter

            return polished_image

        except Exception as e:
            self.logger.warning(f"Final polish failed: {e}")
            return image

    def generate_multiple_variants(
        self,
        video_path: Path,
        analysis: VideoAnalysis,
        output_dir: Path,
        variants: int = 3,
    ) -> list[Path]:
        """
        Generate multiple thumbnail variants for A/B testing

        Args:
            video_path: Source video path
            analysis: AI analysis
            output_dir: Directory to save variants
            variants: Number of variants to generate

        Returns:
            List of paths to generated thumbnails
        """
        generated_paths = []

        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate variants with different timestamps and styles
            timestamps = self._calculate_variant_timestamps(video_path, variants)

            for i, timestamp in enumerate(timestamps):
                # Create modified analysis for this variant
                variant_analysis = analysis.copy(deep=True)
                variant_analysis.thumbnail_info.timestamp_seconds = timestamp

                # Modify headline text for variants
                if i > 0 and analysis.hook_variations:
                    variant_index = (i - 1) % len(analysis.hook_variations)
                    variant_analysis.thumbnail_info.headline_text = (
                        analysis.hook_variations[variant_index]
                    )

                variant_path = output_dir / f"thumbnail_variant_{i + 1}.jpg"

                if self.generate_thumbnail(video_path, variant_analysis, variant_path):
                    generated_paths.append(variant_path)
                    self.logger.info(
                        f"Generated thumbnail variant {i + 1}: {variant_path}"
                    )

            self.logger.info(f"Generated {len(generated_paths)} thumbnail variants")
            return generated_paths

        except Exception as e:
            self.logger.error(f"Error generating thumbnail variants: {e}")
            return generated_paths

    def _calculate_variant_timestamps(
        self, video_path: Path, variants: int
    ) -> list[float]:
        """Calculate optimal timestamps for thumbnail variants"""
        try:
            with VideoFileClip(str(video_path)) as clip:
                duration = clip.duration

                # Calculate timestamps at different points
                timestamps = []
                for i in range(variants):
                    # Distribute across video duration, avoiding very start/end
                    position = 0.2 + (
                        0.6 * i / max(1, variants - 1)
                    )  # 20% to 80% of video
                    timestamp = duration * position
                    timestamps.append(min(timestamp, duration - 0.1))

                return timestamps

        except Exception as e:
            self.logger.warning(f"Error calculating variant timestamps: {e}")
            # Fallback timestamps
            return [5.0, 15.0, 30.0][:variants]

    # ---------------------------------------------------------------------------
    # A/B Testing & Enhanced Optimization Methods
    # ---------------------------------------------------------------------------

    def generate_ab_test_thumbnails(
        self,
        video_path: Path,
        analysis: VideoAnalysisEnhanced,
        output_dir: Path,
        num_variants: int = 3,
    ) -> List[ThumbnailVariant]:
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
            variant_configs = self._generate_variant_configs(
                analysis, num_variants, performance_data
            )

            for i, config in enumerate(variant_configs):
                try:
                    variant_id = (
                        f"variant_{i + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    variant_path = output_dir / f"{variant_id}.jpg"

                    # Generate thumbnail with specific configuration
                    success = self._generate_optimized_thumbnail(
                        video_path, analysis, variant_path, config
                    )

                    if success:
                        variant = ThumbnailVariant(
                            variant_id=variant_id,
                            headline_text=config["headline_text"],
                            timestamp_seconds=config["timestamp_seconds"],
                            text_style=config["text_style"],
                            color_scheme=config["color_scheme"],
                            emotional_tone=config["emotional_tone"],
                        )
                        variants.append(variant)

                        # Store variant configuration for tracking
                        self._store_variant_config(
                            variant_id, config, str(variant_path)
                        )

                        self.logger.info(f"Generated A/B test variant: {variant_id}")

                except Exception as e:
                    self.logger.warning(f"Failed to generate variant {i + 1}: {e}")

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
                "avg_ctr_by_style": {
                    "bold_impact": 0.15,
                    "dramatic_tall": 0.12,
                    "compact_punch": 0.18,
                    "elegant_modern": 0.10,
                    "explosive_caps": 0.16,
                },
                "avg_ctr_by_color": {
                    "high_contrast": 0.14,
                    "warm_bright": 0.16,
                    "cool_dramatic": 0.11,
                    "classic_bold": 0.13,
                    "neon_pop": 0.17,
                },
                "avg_ctr_by_tone": {
                    "exciting": 0.15,
                    "dramatic": 0.13,
                    "mysterious": 0.11,
                    "energetic": 0.17,
                    "professional": 0.09,
                },
                "best_performing_combinations": [
                    {
                        "style": "compact_punch",
                        "color": "neon_pop",
                        "tone": "energetic",
                        "ctr": 0.21,
                    },
                    {
                        "style": "explosive_caps",
                        "color": "warm_bright",
                        "tone": "exciting",
                        "ctr": 0.19,
                    },
                    {
                        "style": "bold_impact",
                        "color": "high_contrast",
                        "tone": "dramatic",
                        "ctr": 0.17,
                    },
                ],
            }

            return performance_data

        except Exception as e:
            self.logger.warning(f"Could not retrieve performance data: {e}")
            return {}

    def _generate_variant_configs(
        self,
        analysis: VideoAnalysisEnhanced,
        num_variants: int,
        performance_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
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

    def _generate_optimized_config(
        self,
        variant_index: int,
        headlines: List[str],
        timestamps: List[float],
        performance_data: Dict[str, Any],
        analysis: VideoAnalysisEnhanced,
    ) -> Dict[str, Any]:
        """Generate an optimized configuration for a variant"""

        # Strategy selection based on variant index
        if variant_index == 0:
            # Control variant - use best performing combination
            if performance_data.get("best_performing_combinations"):
                best = performance_data["best_performing_combinations"][0]
                return {
                    "headline_text": headlines[0] if headlines else "Must Watch!",
                    "timestamp_seconds": timestamps[0] if timestamps else 10.0,
                    "text_style": best["style"],
                    "color_scheme": best["color"],
                    "emotional_tone": best["tone"],
                    "optimization_strategy": "best_known",
                }

        # Experimental variants
        strategies = [
            "high_contrast",
            "emotional_impact",
            "curiosity_gap",
            "trend_following",
        ]
        strategy = strategies[variant_index % len(strategies)]

        if strategy == "high_contrast":
            return {
                "headline_text": headlines[variant_index % len(headlines)]
                if headlines
                else "Amazing!",
                "timestamp_seconds": timestamps[variant_index % len(timestamps)]
                if timestamps
                else 5.0,
                "text_style": "bold_impact",
                "color_scheme": "high_contrast",
                "emotional_tone": "dramatic",
                "optimization_strategy": "high_contrast",
            }

        elif strategy == "emotional_impact":
            return {
                "headline_text": headlines[variant_index % len(headlines)]
                if headlines
                else "Incredible!",
                "timestamp_seconds": timestamps[variant_index % len(timestamps)]
                if timestamps
                else 15.0,
                "text_style": "explosive_caps",
                "color_scheme": "warm_bright",
                "emotional_tone": "exciting",
                "optimization_strategy": "emotional_impact",
            }

        elif strategy == "curiosity_gap":
            # Use more mysterious/intriguing elements
            CURIOSITY_WORDS = {"secret", "hidden", "revealed", "truth", "mystery"}
            curiosity_headlines = [
                h
                for h in headlines
                if any(word in h.lower() for word in CURIOSITY_WORDS)
            ]
            headline = (
                curiosity_headlines[0]
                if curiosity_headlines
                else (headlines[0] if headlines else "You Won't Believe...")
            )

            return {
                "headline_text": headline,
                "timestamp_seconds": timestamps[variant_index % len(timestamps)]
                if timestamps
                else 20.0,
                "text_style": "dramatic_tall",
                "color_scheme": "cool_dramatic",
                "emotional_tone": "mysterious",
                "optimization_strategy": "curiosity_gap",
            }

        else:  # trend_following
            return {
                "headline_text": headlines[variant_index % len(headlines)]
                if headlines
                else "Viral!",
                "timestamp_seconds": timestamps[variant_index % len(timestamps)]
                if timestamps
                else 8.0,
                "text_style": "compact_punch",
                "color_scheme": "neon_pop",
                "emotional_tone": "energetic",
                "optimization_strategy": "trend_following",
            }

    def _calculate_optimal_timestamps(
        self, analysis: VideoAnalysisEnhanced, num_variants: int
    ) -> List[float]:
        """Calculate optimal timestamps for thumbnail variants"""
        timestamps = []

        # Use AI-suggested moments first
        if analysis.key_focus_points:
            focus_times = [fp.timestamp_seconds for fp in analysis.key_focus_points]
            timestamps.extend(focus_times[:num_variants])

        # Add dramatic moments
        if hasattr(analysis, "visual_hook_moment") and analysis.visual_hook_moment:
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
            timestamp = (
                position * estimated_duration * 0.8 + estimated_duration * 0.1
            )  # 10% to 90%
            timestamps.append(timestamp)

        return timestamps[:num_variants]

    def _generate_optimized_thumbnail(
        self,
        video_path: Path,
        analysis: VideoAnalysisEnhanced,
        output_path: Path,
        config: Dict[str, Any],
    ) -> bool:
        """Generate thumbnail with specific optimization configuration"""
        try:
            # Extract frame at specified timestamp
            frame = self._extract_optimal_frame_with_config(video_path, config)
            if frame is None:
                return False

            # Convert to PIL Image
            pil_image = self._cv2_to_pil(frame)

            # Normalize to standard 1280x720 thumbnail dimensions
            pil_image = pil_image.resize((1280, 720), Image.Resampling.LANCZOS)

            # Apply emotional tone adjustments
            enhanced_image = self._apply_emotional_tone(
                pil_image, config["emotional_tone"]
            )

            # Add optimized text overlay
            text_image = self._add_optimized_text(enhanced_image, config)

            # Apply final enhancements
            final_image = self._apply_variant_enhancements(text_image, config)

            # Save thumbnail
            final_image.save(str(output_path), "JPEG", quality=95, optimize=True)

            return True

        except Exception as e:
            self.logger.error(f"Optimized thumbnail generation failed: {e}")
            return False

    def _extract_optimal_frame_with_config(
        self, video_path: Path, config: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Extract frame with configuration-specific optimization"""
        try:
            timestamp = config["timestamp_seconds"]

            with VideoFileClip(str(video_path)) as clip:
                # Ensure timestamp is within bounds
                timestamp = min(timestamp, clip.duration - 0.1)
                timestamp = max(0, timestamp)

                # For high-contrast strategy, find frame with good contrast
                if config.get("optimization_strategy") == "high_contrast":
                    timestamp = self._find_high_contrast_frame(clip, timestamp)

                # Extract frame
                frame = clip.get_frame(timestamp)
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        except Exception as e:
            self.logger.warning(f"Frame extraction failed: {e}")
            return None

    def _find_high_contrast_frame(
        self, clip: VideoFileClip, target_time: float, search_window: float = 2.0
    ) -> float:
        """Find frame with highest contrast around target time"""
        try:
            start_time = max(0, target_time - search_window / 2)
            end_time = min(clip.duration, target_time + search_window / 2)

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
            tone_config = self.emotional_tones.get(
                tone, self.emotional_tones["exciting"]
            )

            # Apply saturation adjustment
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(tone_config["saturation"])

            # Apply contrast adjustment
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(tone_config["contrast"])

            # Apply brightness adjustment
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(tone_config["brightness"])

            return image

        except Exception as e:
            self.logger.warning(f"Emotional tone application failed: {e}")
            return image

    def _add_optimized_text(
        self, image: Image.Image, config: Dict[str, Any]
    ) -> Image.Image:
        """Add optimized text overlay based on configuration"""
        try:
            text = config["headline_text"]
            style_config = self.text_styles.get(
                config["text_style"], self.text_styles["bold_impact"]
            )
            color_config = self.color_schemes.get(
                config["color_scheme"], self.color_schemes["high_contrast"]
            )

            # Create working copy
            img_with_text = image.copy()
            draw = ImageDraw.Draw(img_with_text)

            # Load font with style-specific sizing
            font_path = self.config.get_font_path("BebasNeue-Regular.ttf")
            base_font_size = min(120, image.width // (len(text) // 2 + 5))
            font_size = int(base_font_size * style_config["size_factor"])
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
            if color_config["bg_overlay"] > 0:
                overlay = Image.new(
                    "RGBA", image.size, (0, 0, 0, int(255 * color_config["bg_overlay"]))
                )
                img_with_text = Image.alpha_composite(
                    img_with_text.convert("RGBA"), overlay
                ).convert("RGB")
                draw = ImageDraw.Draw(img_with_text)

            # Draw text with style-specific stroke
            stroke_width = max(3, font_size // 15)

            # Draw stroke
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text(
                            (x + dx, y + dy),
                            text,
                            font=font,
                            fill=color_config["stroke_color"],
                        )

            # Draw main text
            draw.text((x, y), text, font=font, fill=color_config["text_color"])

            return img_with_text

        except Exception as e:
            self.logger.warning(f"Optimized text overlay failed: {e}")
            return image

    def _apply_variant_enhancements(
        self, image: Image.Image, config: Dict[str, Any]
    ) -> Image.Image:
        """Apply variant-specific final enhancements"""
        try:
            strategy = config.get("optimization_strategy", "default")

            if strategy == "high_contrast":
                # Additional contrast boost
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.1)

            elif strategy == "emotional_impact":
                # Slight warmth and saturation boost
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1.15)

            elif strategy == "curiosity_gap":
                # Slight desaturation for mystery
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(0.9)
                # Add subtle vignette
                image = self._add_subtle_vignette(image)

            elif strategy == "trend_following":
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

            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
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

    def _store_variant_config(
        self, variant_id: str, config: Dict[str, Any], file_path: str
    ):
        """Store variant configuration for performance tracking"""
        try:
            config_data = {
                "variant_id": variant_id,
                "config": config,
                "file_path": file_path,
                "created_at": datetime.now().isoformat(),
                "performance_data": None,
            }

            # Store in database or file system
            config_path = Path(file_path).parent / f"{variant_id}_config.json"
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Failed to store variant config: {e}")

    def update_variant_performance(
        self, variant_id: str, performance_metrics: PerformanceMetrics
    ):
        """Update performance metrics for a thumbnail variant"""
        try:
            # Calculate performance score
            performance_score = self._calculate_performance_score(performance_metrics)

            # Store performance data
            self.engagement_db.store_video_metrics(performance_metrics)

            # Update variant with performance data
            # This would integrate with the variant tracking system
            self.logger.info(
                f"Updated performance for variant {variant_id}: score {performance_score:.2f}"
            )

        except Exception as e:
            self.logger.error(f"Failed to update variant performance: {e}")

    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall performance score for a thumbnail"""
        try:
            # Weighted scoring system
            weights = {
                "ctr": 0.4,  # Click-through rate is most important
                "retention": 0.3,  # Viewer retention
                "engagement": 0.2,  # Likes, comments, shares
                "completion": 0.1,  # Video completion rate
            }

            # Normalize metrics to 0-100 scale
            ctr_score = min(
                metrics.click_through_rate * 1000, 100
            )  # CTR is typically small
            retention_score = metrics.watch_time_percentage

            engagement_rate = 0
            if metrics.views > 0:
                total_engagement = metrics.likes + metrics.comments + metrics.shares
                engagement_rate = (total_engagement / metrics.views) * 100
                engagement_rate = min(engagement_rate, 100)

            completion_score = metrics.watch_time_percentage  # Proxy for completion

            # Calculate weighted score
            total_score = (
                ctr_score * weights["ctr"]
                + retention_score * weights["retention"]
                + engagement_rate * weights["engagement"]
                + completion_score * weights["completion"]
            )

            return min(total_score, 100)

        except Exception as e:
            self.logger.warning(f"Performance score calculation failed: {e}")
            return 50.0  # Default average score

    def get_best_performing_variant(
        self, variants: List[ThumbnailVariant]
    ) -> Optional[ThumbnailVariant]:
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
