"""
AI-Powered Cinematic Editor - Virtual Editor for Dynamic Video Enhancement
Identifies key focus points for camera movements, suggests speed ramps, and creates cinematic effects.
"""

import logging
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
from moviepy import VideoFileClip, vfx
from dataclasses import dataclass

from src.config.settings import get_config
from src.models import VideoAnalysisEnhanced, CameraMovement, FocusPoint, SpeedEffect
from src.utils.gpu_memory_manager import GPUMemoryManager


@dataclass
class SceneAnalysis:
    """Analysis of a video scene for cinematic enhancement"""
    timestamp: float
    scene_type: str  # "action", "dialogue", "transition", "climax"
    motion_intensity: float  # 0.0 to 1.0
    visual_complexity: float  # 0.0 to 1.0
    audio_energy: float  # 0.0 to 1.0
    emotional_weight: float  # 0.0 to 1.0
    key_objects: List[Dict[str, Any]]  # Detected objects and their positions
    composition_score: float  # How well-composed the frame is


class CinematicEditor:
    """
    AI-powered virtual editor that analyzes video content and suggests
    cinematic enhancements including dynamic camera movements and speed ramps.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.gpu_manager = GPUMemoryManager(max_vram_usage=0.6)  # Increased for video processing
        
        # Cinematic parameters
        self.min_scene_duration = 2.0  # Minimum scene duration for analysis
        self.motion_threshold = 0.3  # Threshold for detecting significant motion
        self.composition_weights = {
            'rule_of_thirds': 0.3,
            'symmetry': 0.2,
            'leading_lines': 0.2,
            'contrast': 0.3
        }
    
    def analyze_video_cinematically(self, video_path: Path, 
                                   analysis: VideoAnalysisEnhanced) -> VideoAnalysisEnhanced:
        """
        Perform comprehensive cinematic analysis and enhance the analysis with suggestions
        
        Args:
            video_path: Path to video file
            analysis: Current video analysis to enhance
            
        Returns:
            Enhanced analysis with cinematic suggestions
        """
        try:
            self.logger.info("Starting cinematic analysis...")
            
            with VideoFileClip(str(video_path)) as clip:
                # Analyze scenes for cinematic opportunities
                scenes = self._analyze_scenes(clip)
                
                # Generate camera movement suggestions
                camera_movements = self._suggest_camera_movements(scenes, clip.duration)
                analysis.camera_movements = camera_movements
                
                # Generate speed effect suggestions
                speed_effects = self._suggest_speed_effects(scenes)
                analysis.speed_effects.extend(speed_effects)
                
                # Identify dynamic focus points
                dynamic_focus_points = self._identify_dynamic_focus_points(scenes)
                analysis.dynamic_focus_points = dynamic_focus_points
                
                # Generate cinematic transitions
                transitions = self._suggest_cinematic_transitions(scenes)
                analysis.cinematic_transitions = transitions
                
            self.logger.info(f"Cinematic analysis complete: {len(camera_movements)} movements, "
                           f"{len(speed_effects)} speed effects suggested")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Cinematic analysis failed: {e}")
            return analysis
    
    def _analyze_scenes(self, clip: VideoFileClip) -> List[SceneAnalysis]:
        """Analyze video scenes for cinematic opportunities"""
        scenes = []
        duration = clip.duration
        sample_interval = 1.0  # Analyze every second
        
        try:
            for t in np.arange(0, duration, sample_interval):
                if t >= duration:
                    break
                
                # Extract frame for analysis
                frame = clip.get_frame(t)
                
                # Analyze frame composition and content
                scene = self._analyze_frame(frame, t)
                scenes.append(scene)
                
                # Log progress occasionally
                if len(scenes) % 30 == 0:
                    self.logger.debug(f"Analyzed {len(scenes)} scenes...")
        
        except Exception as e:
            self.logger.warning(f"Scene analysis error at {t}s: {e}")
        
        self.logger.info(f"Analyzed {len(scenes)} scenes for cinematic opportunities")
        return scenes
    
    def _analyze_frame(self, frame: np.ndarray, timestamp: float) -> SceneAnalysis:
        """Analyze a single frame for cinematic properties"""
        height, width = frame.shape[:2]
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        # Calculate motion intensity (simplified - would need previous frame for accurate motion)
        motion_intensity = self._estimate_motion_from_frame(gray)
        
        # Calculate visual complexity
        visual_complexity = self._calculate_visual_complexity(gray)
        
        # Estimate audio energy (would be enhanced with actual audio analysis)
        audio_energy = min(motion_intensity * 1.2, 1.0)  # Correlation approximation
        
        # Calculate emotional weight based on color and contrast
        emotional_weight = self._calculate_emotional_weight(hsv, gray)
        
        # Detect key objects/regions
        key_objects = self._detect_key_regions(frame, gray)
        
        # Calculate composition score
        composition_score = self._calculate_composition_score(gray, key_objects)
        
        # Classify scene type based on analysis
        scene_type = self._classify_scene_type(motion_intensity, visual_complexity, emotional_weight)
        
        return SceneAnalysis(
            timestamp=timestamp,
            scene_type=scene_type,
            motion_intensity=motion_intensity,
            visual_complexity=visual_complexity,
            audio_energy=audio_energy,
            emotional_weight=emotional_weight,
            key_objects=key_objects,
            composition_score=composition_score
        )
    
    def _estimate_motion_from_frame(self, gray: np.ndarray) -> float:
        """Estimate motion intensity from frame characteristics"""
        # Calculate edge density as motion indicator
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate variance as another motion indicator
        variance = np.var(gray) / (255 * 255)
        
        # Combine metrics
        motion_score = min((edge_density * 3 + variance * 2) / 2, 1.0)
        return motion_score
    
    def _calculate_visual_complexity(self, gray: np.ndarray) -> float:
        """Calculate visual complexity of the frame"""
        # Calculate texture using local binary patterns approximation
        texture = cv2.Laplacian(gray, cv2.CV_64F).var()
        normalized_texture = min(texture / 1000, 1.0)  # Normalize
        
        # Calculate contrast
        contrast = gray.std() / 255
        
        # Combine metrics
        complexity = (normalized_texture * 0.6 + contrast * 0.4)
        return min(complexity, 1.0)
    
    def _calculate_emotional_weight(self, hsv: np.ndarray, gray: np.ndarray) -> float:
        """Calculate emotional weight based on color and contrast"""
        # Analyze saturation for emotional intensity
        saturation = hsv[:, :, 1].mean() / 255
        
        # Analyze brightness distribution
        brightness = hsv[:, :, 2].mean() / 255
        
        # Calculate contrast for drama
        contrast = gray.std() / 255
        
        # Combine for emotional weight
        emotional_weight = (saturation * 0.4 + contrast * 0.4 + (abs(brightness - 0.5) * 2) * 0.2)
        return min(emotional_weight, 1.0)
    
    def _detect_key_regions(self, frame: np.ndarray, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect key regions/objects in the frame"""
        key_objects = []
        height, width = gray.shape
        
        try:
            # Detect corners/interest points
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=10, qualityLevel=0.01, minDistance=30)
            
            if corners is not None:
                for corner in corners:
                    x, y = corner.ravel()
                    # Normalize coordinates
                    normalized_x = x / width
                    normalized_y = y / height
                    
                    key_objects.append({
                        'type': 'interest_point',
                        'position': (normalized_x, normalized_y),
                        'strength': 1.0  # Would be calculated from feature response
                    })
            
            # Detect high-contrast regions
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find significant contours
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > (height * width * 0.01):  # At least 1% of frame
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = (x + w/2) / width
                    center_y = (y + h/2) / height
                    
                    key_objects.append({
                        'type': 'high_contrast_region',
                        'position': (center_x, center_y),
                        'size': (w/width, h/height),
                        'strength': min(area / (height * width), 1.0)
                    })
        
        except Exception as e:
            self.logger.debug(f"Key region detection error: {e}")
        
        # Sort by strength and limit to top objects
        key_objects.sort(key=lambda x: x.get('strength', 0), reverse=True)
        return key_objects[:5]  # Top 5 objects
    
    def _calculate_composition_score(self, gray: np.ndarray, key_objects: List[Dict[str, Any]]) -> float:
        """Calculate how well-composed the frame is"""
        height, width = gray.shape
        score = 0.0
        
        # Rule of thirds scoring
        third_lines_x = [width/3, 2*width/3]
        third_lines_y = [height/3, 2*height/3]
        
        for obj in key_objects:
            if 'position' in obj:
                x, y = obj['position']
                pixel_x, pixel_y = x * width, y * height
                
                # Check proximity to rule of thirds lines
                min_dist_x = min(abs(pixel_x - line) for line in third_lines_x)
                min_dist_y = min(abs(pixel_y - line) for line in third_lines_y)
                
                # Score based on proximity (closer = better)
                thirds_score = max(0, 1 - (min_dist_x + min_dist_y) / (width + height))
                score += thirds_score * obj.get('strength', 1.0) * self.composition_weights['rule_of_thirds']
        
        # Symmetry scoring (simplified)
        center_mass_x = sum(obj['position'][0] * obj.get('strength', 1.0) for obj in key_objects if 'position' in obj)
        if key_objects:
            center_mass_x /= len(key_objects)
            symmetry_score = 1 - abs(center_mass_x - 0.5) * 2  # Distance from center
            score += symmetry_score * self.composition_weights['symmetry']
        
        # Contrast scoring
        contrast_score = gray.std() / 255
        score += contrast_score * self.composition_weights['contrast']
        
        return min(score, 1.0)
    
    def _classify_scene_type(self, motion: float, complexity: float, emotion: float) -> str:
        """Classify scene type based on analyzed metrics"""
        if motion > 0.7 and complexity > 0.6:
            return "action"
        elif emotion > 0.7 and motion < 0.3:
            return "dramatic"
        elif motion > 0.4 and complexity > 0.4:
            return "dynamic"
        elif emotion < 0.3 and motion < 0.3:
            return "calm"
        else:
            return "general"
    
    def _suggest_camera_movements(self, scenes: List[SceneAnalysis], 
                                 duration: float) -> List[CameraMovement]:
        """Generate camera movement suggestions based on scene analysis"""
        movements = []
        
        # Group scenes into segments for movement planning
        segments = self._group_scenes_into_segments(scenes)
        
        for segment in segments:
            if len(segment) < 2:
                continue
                
            start_scene = segment[0]
            end_scene = segment[-1]
            
            # Determine appropriate movement based on scene characteristics
            movement = self._determine_camera_movement(start_scene, end_scene, segment)
            if movement:
                movements.append(movement)
        
        # Add dramatic moments with zoom effects
        for scene in scenes:
            if scene.emotional_weight > 0.8 and scene.scene_type in ["dramatic", "action"]:
                zoom_movement = self._create_dramatic_zoom(scene)
                if zoom_movement:
                    movements.append(zoom_movement)
        
        # Sort movements by start time
        movements.sort(key=lambda x: x.start_time)
        
        self.logger.debug(f"Generated {len(movements)} camera movement suggestions")
        return movements
    
    def _group_scenes_into_segments(self, scenes: List[SceneAnalysis]) -> List[List[SceneAnalysis]]:
        """Group consecutive scenes into movement segments"""
        if not scenes:
            return []
        
        segments = []
        current_segment = [scenes[0]]
        
        for i in range(1, len(scenes)):
            prev_scene = scenes[i-1]
            current_scene = scenes[i]
            
            # Start new segment if scene type changes significantly or time gap is large
            time_gap = current_scene.timestamp - prev_scene.timestamp
            scene_change = prev_scene.scene_type != current_scene.scene_type
            
            if time_gap > 5.0 or scene_change:
                if len(current_segment) >= 2:
                    segments.append(current_segment)
                current_segment = [current_scene]
            else:
                current_segment.append(current_scene)
        
        # Add final segment
        if len(current_segment) >= 2:
            segments.append(current_segment)
        
        return segments
    
    def _determine_camera_movement(self, start_scene: SceneAnalysis, 
                                  end_scene: SceneAnalysis,
                                  segment: List[SceneAnalysis]) -> Optional[CameraMovement]:
        """Determine appropriate camera movement for a scene segment"""
        # Calculate segment characteristics
        avg_motion = np.mean([s.motion_intensity for s in segment])
        avg_emotion = np.mean([s.emotional_weight for s in segment])
        duration = end_scene.timestamp - start_scene.timestamp
        
        # Skip very short segments
        if duration < self.min_scene_duration:
            return None
        
        # Find key objects for movement targeting
        start_objects = start_scene.key_objects
        end_objects = end_scene.key_objects
        
        if not start_objects or not end_objects:
            return None
        
        # Use strongest objects as focal points
        start_focus = start_objects[0]['position']
        end_focus = end_objects[0]['position']
        
        # Determine movement type based on scene characteristics
        if avg_motion > 0.6:
            # High motion: dynamic pan/zoom
            movement_type = "pan_zoom"
            zoom_factor = 1.2 + (avg_emotion * 0.3)  # More dramatic zoom for emotional scenes
            easing = "ease_in_out"
        elif avg_emotion > 0.7:
            # High emotion: slow zoom
            movement_type = "zoom"
            zoom_factor = 1.1 + (avg_emotion * 0.2)
            easing = "ease_in"
        else:
            # General: subtle pan
            movement_type = "pan"
            zoom_factor = 1.0
            easing = "linear"
        
        # Calculate intensity based on scene energy with minimum threshold
        intensity = max(min((avg_motion + avg_emotion) / 2, 1.0), 0.1)
        
        return CameraMovement(
            start_time=start_scene.timestamp,
            end_time=end_scene.timestamp,
            movement_type=movement_type,
            start_position=start_focus,
            end_position=end_focus,
            zoom_factor=zoom_factor,
            easing=easing,
            intensity=intensity
        )
    
    def _create_dramatic_zoom(self, scene: SceneAnalysis) -> Optional[CameraMovement]:
        """Create dramatic zoom effect for high-impact moments"""
        if not scene.key_objects:
            return None
        
        # Use strongest object as zoom target
        target_object = scene.key_objects[0]
        target_position = target_object['position']
        
        # Create zoom-in effect
        zoom_duration = 2.0  # 2-second zoom
        
        return CameraMovement(
            start_time=max(0, scene.timestamp - zoom_duration/2),
            end_time=scene.timestamp + zoom_duration/2,
            movement_type="zoom",
            start_position=target_position,
            end_position=target_position,
            zoom_factor=1.5 + (scene.emotional_weight * 0.5),
            easing="ease_in_out",
            intensity=max(scene.emotional_weight, 0.1)  # Ensure minimum intensity
        )
    
    def _suggest_speed_effects(self, scenes: List[SceneAnalysis]) -> List[SpeedEffect]:
        """Generate speed effect suggestions for dramatic impact"""
        speed_effects = []
        
        for i, scene in enumerate(scenes):
            # Slow motion for dramatic moments
            if scene.emotional_weight > 0.8 and scene.scene_type in ["dramatic", "action"]:
                slow_mo_effect = SpeedEffect(
                    start_seconds=scene.timestamp,
                    end_seconds=min(scene.timestamp + 2.0, scenes[-1].timestamp),
                    speed_factor=0.5 + (scene.emotional_weight * 0.3),  # 0.5x to 0.8x speed
                    effect_type="slow_motion"
                )
                speed_effects.append(slow_mo_effect)
            
            # Speed up for transitional/buildup moments
            elif scene.motion_intensity > 0.7 and scene.scene_type == "action":
                if i < len(scenes) - 1:
                    next_scene = scenes[i + 1]
                    if next_scene.emotional_weight > scene.emotional_weight:
                        # Building up to something more dramatic
                        speedup_effect = SpeedEffect(
                            start_seconds=scene.timestamp,
                            end_seconds=next_scene.timestamp,
                            speed_factor=1.2 + (scene.motion_intensity * 0.3),  # 1.2x to 1.5x speed
                            effect_type="speed_up"
                        )
                        speed_effects.append(speedup_effect)
        
        self.logger.debug(f"Generated {len(speed_effects)} speed effect suggestions")
        return speed_effects
    
    def _identify_dynamic_focus_points(self, scenes: List[SceneAnalysis]) -> List[FocusPoint]:
        """Identify dynamic focus points that change over time"""
        focus_points = []
        
        for scene in scenes:
            for obj in scene.key_objects:
                if obj.get('strength', 0) > 0.7:  # Only strong focal points
                    focus_point = FocusPoint(
                        x=obj['position'][0],
                        y=obj['position'][1],
                        timestamp_seconds=scene.timestamp,
                        description=f"Dynamic focus - {obj['type']} (strength: {obj.get('strength', 0):.2f})"
                    )
                    focus_points.append(focus_point)
        
        self.logger.debug(f"Identified {len(focus_points)} dynamic focus points")
        return focus_points
    
    def _suggest_cinematic_transitions(self, scenes: List[SceneAnalysis]) -> List[Dict[str, Any]]:
        """Suggest cinematic transitions between scenes"""
        transitions = []
        
        for i in range(len(scenes) - 1):
            current_scene = scenes[i]
            next_scene = scenes[i + 1]
            
            # Determine transition type based on scene change
            time_gap = next_scene.timestamp - current_scene.timestamp
            emotion_change = abs(next_scene.emotional_weight - current_scene.emotional_weight)
            motion_change = abs(next_scene.motion_intensity - current_scene.motion_intensity)
            
            if time_gap > 2.0 and (emotion_change > 0.4 or motion_change > 0.4):
                # Significant scene change - suggest transition
                transition_type = self._determine_transition_type(current_scene, next_scene)
                
                transition = {
                    'start_time': current_scene.timestamp,
                    'end_time': next_scene.timestamp,
                    'type': transition_type,
                    'duration': min(1.0, time_gap / 2),  # Max 1 second transition
                    'intensity': (emotion_change + motion_change) / 2
                }
                transitions.append(transition)
        
        self.logger.debug(f"Suggested {len(transitions)} cinematic transitions")
        return transitions
    
    def _determine_transition_type(self, current_scene: SceneAnalysis, 
                                  next_scene: SceneAnalysis) -> str:
        """Determine appropriate transition type between scenes"""
        emotion_change = next_scene.emotional_weight - current_scene.emotional_weight
        motion_change = next_scene.motion_intensity - current_scene.motion_intensity
        
        if emotion_change > 0.3:
            return "fade_in" if motion_change > 0 else "dissolve"
        elif emotion_change < -0.3:
            return "fade_out" if motion_change < 0 else "cut"
        elif motion_change > 0.4:
            return "zoom_transition"
        else:
            return "crossfade"