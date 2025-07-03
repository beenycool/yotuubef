"""
Utility functions for video processing and validation.
Provides reusable functions for segment selection, validation, and data processing.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from src.models import VideoAnalysis, VideoSegment
from src.config.settings import get_config


def select_and_validate_segments(analysis: VideoAnalysis, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Takes the raw analysis and validates the segments against the video duration
    and target short duration, returning a list of valid segment timings.
    
    Args:
        analysis: VideoAnalysis object with validated segments
        config: Configuration dictionary with video constraints
    
    Returns:
        List of valid segment dictionaries with timing information
    """
    logger = logging.getLogger(__name__)
    valid_segments = []
    
    try:
        # Get configuration values
        max_short_duration = config.get('video', {}).get('max_short_duration_seconds', 60)
        min_short_duration = config.get('video', {}).get('min_short_duration_seconds', 15)
        
        # Use the best segment first, then other segments
        segments_to_check = [analysis.best_segment] + analysis.segments
        
        for segment in segments_to_check:
            # Calculate segment duration
            segment_duration = segment.end_seconds - segment.start_seconds
            
            # Validate segment timing
            if segment.start_seconds < 0:
                logger.warning(f"Segment has negative start time: {segment.start_seconds}")
                continue
            
            if segment.end_seconds <= segment.start_seconds:
                logger.warning(f"Segment has invalid end time: {segment.end_seconds} <= {segment.start_seconds}")
                continue
            
            # Check duration constraints with intelligent handling for very short videos
            if segment_duration < min_short_duration:
                # For very short segments, check if this represents most of the available content
                if hasattr(segment, 'end_seconds') and segment.end_seconds > 0:
                    # If segment represents >70% of available content and is >2 seconds, accept it anyway
                    coverage_ratio = segment_duration / segment.end_seconds
                    if coverage_ratio > 0.7 and segment_duration > 2.0:
                        logger.info(f"Accepting short segment due to high content coverage: {segment_duration:.1f}s "
                                  f"({coverage_ratio:.1%} of content)")
                    else:
                        logger.warning(f"Segment too short: {segment_duration:.1f}s < {min_short_duration:.1f}s minimum")
                        continue
                else:
                    logger.warning(f"Segment too short: {segment_duration:.1f}s < {min_short_duration:.1f}s minimum")
                    continue
            
            if segment_duration > max_short_duration:
                logger.warning(f"Segment too long: {segment_duration}s > {max_short_duration}s maximum")
                # Truncate segment to max duration
                segment_data = {
                    'start_seconds': segment.start_seconds,
                    'end_seconds': segment.start_seconds + max_short_duration,
                    'duration_seconds': max_short_duration,
                    'reason': f"{segment.reason} (truncated to {max_short_duration}s)",
                    'original_end': segment.end_seconds
                }
            else:
                segment_data = {
                    'start_seconds': segment.start_seconds,
                    'end_seconds': segment.end_seconds,
                    'duration_seconds': segment_duration,
                    'reason': segment.reason,
                    'original_end': segment.end_seconds
                }
            
            # Check if we already have this segment (avoid duplicates)
            is_duplicate = any(
                abs(seg['start_seconds'] - segment_data['start_seconds']) < 1.0 and
                abs(seg['end_seconds'] - segment_data['end_seconds']) < 1.0
                for seg in valid_segments
            )
            
            if not is_duplicate:
                valid_segments.append(segment_data)
                logger.info(f"Valid segment: {segment_data['start_seconds']:.1f}s - {segment_data['end_seconds']:.1f}s ({segment_data['duration_seconds']:.1f}s)")
        
        # Sort segments by start time
        valid_segments.sort(key=lambda x: x['start_seconds'])
        
        logger.info(f"Found {len(valid_segments)} valid segments out of {len(segments_to_check)} analyzed")
        return valid_segments
        
    except Exception as e:
        logger.error(f"Error validating segments: {e}")
        return []


def validate_file_paths(video_path: Path, output_path: Path) -> Tuple[bool, str]:
    """
    Validate input and output file paths
    
    Args:
        video_path: Path to input video file
        output_path: Path for output video file
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check input file exists
        if not video_path.exists():
            return False, f"Input video file does not exist: {video_path}"
        
        # Check input file is readable
        if not video_path.is_file():
            return False, f"Input path is not a file: {video_path}"
        
        # Check output directory can be created
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check we can write to output location
        if output_path.exists() and not output_path.is_file():
            return False, f"Output path exists but is not a file: {output_path}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Path validation error: {e}"


def get_safe_filename(title: str, max_length: int = 50) -> str:
    """
    Convert a title to a safe filename
    
    Args:
        title: Original title string
        max_length: Maximum filename length
    
    Returns:
        Safe filename string
    """
    import re
    
    # Remove or replace unsafe characters
    safe_title = re.sub(r'[<>:"/\\|?*]', '', title)
    safe_title = re.sub(r'\s+', '_', safe_title.strip())
    
    # Limit length
    if len(safe_title) > max_length:
        safe_title = safe_title[:max_length].rstrip('_')
    
    # Ensure not empty
    if not safe_title:
        safe_title = "video"
    
    return safe_title


def calculate_video_metrics(analysis: VideoAnalysis) -> Dict[str, Any]:
    """
    Calculate useful metrics from video analysis
    
    Args:
        analysis: VideoAnalysis object
    
    Returns:
        Dictionary of calculated metrics
    """
    try:
        metrics = {
            'total_segments': len(analysis.segments),
            'has_text_overlays': len(analysis.text_overlays) > 0,
            'has_narrative': len(analysis.narrative_script_segments) > 0,
            'has_visual_effects': len(analysis.visual_cues) > 0,
            'has_speed_effects': len(analysis.speed_effects) > 0,
            'complexity_score': 0
        }
        
        # Calculate complexity score
        complexity = 0
        complexity += len(analysis.text_overlays) * 2
        complexity += len(analysis.narrative_script_segments) * 3
        complexity += len(analysis.visual_cues) * 2
        complexity += len(analysis.speed_effects) * 4
        complexity += len(analysis.key_focus_points)
        
        metrics['complexity_score'] = complexity
        
        # Determine processing priority (higher complexity = lower priority for resource management)
        if complexity <= 10:
            metrics['processing_priority'] = 'high'
        elif complexity <= 25:
            metrics['processing_priority'] = 'medium'
        else:
            metrics['processing_priority'] = 'low'
        
        return metrics
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Error calculating video metrics: {e}")
        return {
            'total_segments': 0,
            'has_text_overlays': False,
            'has_narrative': False,
            'has_visual_effects': False,
            'has_speed_effects': False,
            'complexity_score': 0,
            'processing_priority': 'medium'
        }


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted duration string (e.g., "1:23" or "45s")
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    else:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"


def validate_analysis_completeness(analysis: VideoAnalysis) -> Tuple[bool, List[str]]:
    """
    Validate that the analysis has all required components for video processing
    
    Args:
        analysis: VideoAnalysis object to validate
    
    Returns:
        Tuple of (is_complete, list_of_missing_items)
    """
    missing_items = []
    
    try:
        # Check essential fields
        if not analysis.suggested_title or len(analysis.suggested_title.strip()) == 0:
            missing_items.append("suggested_title")
        
        if not analysis.segments or len(analysis.segments) == 0:
            missing_items.append("video_segments")
        
        if not analysis.hashtags or len(analysis.hashtags) == 0:
            missing_items.append("hashtags")
        
        # Check if we have either original audio or TTS content
        has_audio_content = (
            analysis.original_audio_is_key or 
            len(analysis.narrative_script_segments) > 0
        )
        
        if not has_audio_content:
            missing_items.append("audio_content")
        
        # Check basic metadata
        if not analysis.mood:
            missing_items.append("mood")
        
        is_complete = len(missing_items) == 0
        return is_complete, missing_items
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error validating analysis completeness: {e}")
        return False, ["validation_error"]