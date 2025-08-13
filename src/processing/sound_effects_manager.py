"""
Enhanced Sound Effects Manager for YouTube video generation system.
Provides intelligent sound effect mapping, directory-based search, and automated testing.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
from moviepy.audio.io.AudioFileClip import AudioFileClip

from src.config.settings import get_config


@dataclass
class SoundEffectFile:
    """Represents a sound effect file with metadata"""
    path: Path
    name: str
    category: str
    duration: Optional[float] = None
    volume_peak: Optional[float] = None
    aliases: List[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


class SoundEffectsManager:
    """
    Advanced sound effects management with intelligent mapping and fallback strategies.
    Follows the directory structure defined in sound_effects/README.md
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.sound_effects_dir = self.config.paths.sound_effects_folder
        
        # Enhanced category-based mapping with intelligent fallbacks
        self.category_mapping = {
            # Primary categories from README.md
            'impact': {
                'keywords': ['impact', 'hit', 'thud', 'bang', 'crash', 'slam', 'smash', 'punch'],
                'fallbacks': ['mechanical', 'dramatic']
            },
            'transition': {
                'keywords': ['whoosh', 'swoosh', 'zip', 'swish', 'wind', 'air', 'move', 'fast'],
                'fallbacks': ['mechanical', 'impact']
            },
            'liquid': {
                'keywords': ['splash', 'pour', 'glug', 'water', 'liquid', 'drop', 'drip', 'bubble'],
                'fallbacks': ['mechanical', 'impact']
            },
            'mechanical': {
                'keywords': ['click', 'pop', 'snap', 'tick', 'beep', 'button', 'switch', 'gear'],
                'fallbacks': ['notification', 'impact']
            },
            'notification': {
                'keywords': ['ding', 'chime', 'bell', 'alert', 'notification', 'ping', 'message'],
                'fallbacks': ['mechanical', 'transition']
            },
            'dramatic': {
                'keywords': ['boom', 'thunder', 'rumble', 'tension', 'dramatic', 'power', 'energy'],
                'fallbacks': ['impact', 'transition']
            }
        }
        
        # Sound effect aliases and alternatives
        self.sound_aliases = {
            'whoosh': ['swoosh', 'swish', 'wind', 'air'],
            'splash': ['water', 'liquid', 'pour', 'glug'],
            'pop': ['click', 'snap', 'bubble'],
            'impact': ['hit', 'thud', 'bang', 'crash'],
            'ding': ['bell', 'chime', 'notification', 'alert'],
            'boom': ['thunder', 'explosion', 'dramatic'],
            'click': ['pop', 'snap', 'button'],
            'pour': ['liquid', 'water', 'stream'],
            'tick': ['clock', 'time', 'timer'],
            'beep': ['electronic', 'digital', 'tech']
        }
        
        # Cache for discovered sound effects
        self._sound_cache: Dict[str, List[SoundEffectFile]] = {}
        self._last_scan_time = 0
        
        # Initialize by scanning directory
        self.refresh_sound_effects_cache()
        
        # Gen Z specific sound effects library
        self.gen_z_effects = {
            "vine_boom": {
                "keywords": ["vine boom", "boom", "impact", "dramatic"],
                "fallbacks": ["boom", "impact", "dramatic"],
                "description": "Classic Vine boom sound effect"
            },
            "ohio_sound": {
                "keywords": ["ohio", "weird", "unusual", "funny"],
                "fallbacks": ["funny", "weird", "mechanical"],
                "description": "Ohio meme sound effect"
            },
            "sad_violin": {
                "keywords": ["sad violin", "emotional", "dramatic", "sad"],
                "fallbacks": ["dramatic", "emotional"],
                "description": "Sad violin for emotional moments"
            },
            "epic_sax": {
                "keywords": ["epic sax", "saxophone", "epic", "dramatic"],
                "fallbacks": ["dramatic", "epic"],
                "description": "Epic saxophone for dramatic moments"
            },
            "cricket_chirp": {
                "keywords": ["cricket", "awkward", "silence", "funny"],
                "fallbacks": ["funny", "awkward"],
                "description": "Cricket chirping for awkward moments"
            },
            "record_scratch": {
                "keywords": ["record scratch", "vinyl", "stop", "interrupt"],
                "fallbacks": ["mechanical", "transition"],
                "description": "Record scratch for interruptions"
            }
        }
    
    def refresh_sound_effects_cache(self) -> None:
        """Scan sound effects directory and build cache"""
        try:
            if not self.sound_effects_dir.exists():
                self.logger.warning(f"Sound effects directory not found: {self.sound_effects_dir}")
                return
            
            self._sound_cache.clear()
            total_effects = 0
            
            # Scan each category directory
            for category, info in self.category_mapping.items():
                category_dir = self.sound_effects_dir / category
                category_files = []
                
                if category_dir.exists():
                    # Find all sound files in category
                    for pattern in ['*.wav', '*.mp3', '*.ogg', '*.m4a']:
                        for file_path in category_dir.glob(pattern):
                            if file_path.is_file():
                                sound_effect = SoundEffectFile(
                                    path=file_path,
                                    name=file_path.stem.lower(),
                                    category=category,
                                    aliases=self._generate_aliases(file_path.stem.lower())
                                )
                                category_files.append(sound_effect)
                                total_effects += 1
                
                self._sound_cache[category] = category_files
                self.logger.debug(f"Found {len(category_files)} effects in {category} category")
            
            # Also scan root directory for uncategorized effects
            root_files = []
            for pattern in ['*.wav', '*.mp3', '*.ogg', '*.m4a']:
                for file_path in self.sound_effects_dir.glob(pattern):
                    if file_path.is_file():
                        # Try to categorize based on filename
                        category = self._categorize_by_filename(file_path.stem.lower())
                        sound_effect = SoundEffectFile(
                            path=file_path,
                            name=file_path.stem.lower(),
                            category=category or 'uncategorized',
                            aliases=self._generate_aliases(file_path.stem.lower())
                        )
                        root_files.append(sound_effect)
                        total_effects += 1
            
            if root_files:
                self._sound_cache['uncategorized'] = root_files
            
            self.logger.info(f"Sound effects cache refreshed: {total_effects} effects across {len(self._sound_cache)} categories")
            
        except Exception as e:
            self.logger.error(f"Error refreshing sound effects cache: {e}")
    
    def _generate_aliases(self, filename: str) -> List[str]:
        """Generate aliases for a sound effect based on filename and known mappings"""
        aliases = []
        
        # Add direct aliases from mapping
        for primary, alias_list in self.sound_aliases.items():
            if primary in filename:
                aliases.extend(alias_list)
            elif filename in alias_list:
                aliases.append(primary)
                aliases.extend([a for a in alias_list if a != filename])
        
        # Add word variations
        if '_' in filename:
            aliases.extend(filename.split('_'))
        if '-' in filename:
            aliases.extend(filename.split('-'))
        
        return list(set(aliases))
    
    def _categorize_by_filename(self, filename: str) -> Optional[str]:
        """Attempt to categorize a sound effect by its filename"""
        for category, info in self.category_mapping.items():
            for keyword in info['keywords']:
                if keyword in filename:
                    return category
        return None
    
    def find_sound_effect(self, effect_name: str, preferred_category: Optional[str] = None) -> Optional[Path]:
        """
        Find the best matching sound effect file with intelligent fallback strategy.
        
        Args:
            effect_name: Name of the effect to find
            preferred_category: Preferred category to search first
            
        Returns:
            Path to the best matching sound effect file, or None if not found
        """
        effect_name = effect_name.lower().strip()
        
        # Strategy 1: Direct match in preferred category
        if preferred_category and preferred_category in self._sound_cache:
            direct_match = self._find_in_category(effect_name, preferred_category, exact=True)
            if direct_match:
                self.logger.debug(f"Found direct match for '{effect_name}' in {preferred_category}: {direct_match}")
                return direct_match
        
        # Strategy 2: Direct match in any category
        for category in self._sound_cache:
            direct_match = self._find_in_category(effect_name, category, exact=True)
            if direct_match:
                self.logger.debug(f"Found direct match for '{effect_name}' in {category}: {direct_match}")
                return direct_match
        
        # Strategy 3: Alias match in preferred category
        if preferred_category and preferred_category in self._sound_cache:
            alias_match = self._find_in_category(effect_name, preferred_category, exact=False)
            if alias_match:
                self.logger.debug(f"Found alias match for '{effect_name}' in {preferred_category}: {alias_match}")
                return alias_match
        
        # Strategy 4: Category-based intelligent search
        suggested_category = self._suggest_category(effect_name)
        if suggested_category and suggested_category != preferred_category:
            category_match = self._find_in_category(effect_name, suggested_category, exact=False)
            if category_match:
                self.logger.debug(f"Found category match for '{effect_name}' in {suggested_category}: {category_match}")
                return category_match
        
        # Strategy 5: Partial match across all categories
        for category in self._sound_cache:
            partial_match = self._find_partial_match(effect_name, category)
            if partial_match:
                self.logger.debug(f"Found partial match for '{effect_name}' in {category}: {partial_match}")
                return partial_match
        
        # Strategy 6: Fallback to similar sounding effects
        fallback_match = self._find_fallback_effect(effect_name)
        if fallback_match:
            self.logger.info(f"Using fallback effect for '{effect_name}': {fallback_match}")
            return fallback_match
        
        self.logger.warning(f"No sound effect found for: {effect_name}")
        return None
    
    def _find_in_category(self, effect_name: str, category: str, exact: bool = True) -> Optional[Path]:
        """Find effect in specific category"""
        if category not in self._sound_cache:
            return None
        
        for sound_effect in self._sound_cache[category]:
            if exact:
                if sound_effect.name == effect_name:
                    return sound_effect.path
            else:
                # Check name and aliases
                if (effect_name == sound_effect.name or 
                    effect_name in sound_effect.aliases or
                    any(alias in sound_effect.name for alias in [effect_name])):
                    return sound_effect.path
        
        return None
    
    def _find_partial_match(self, effect_name: str, category: str) -> Optional[Path]:
        """Find partial matches in category"""
        if category not in self._sound_cache:
            return None
        
        # Look for partial matches
        for sound_effect in self._sound_cache[category]:
            if (effect_name in sound_effect.name or 
                sound_effect.name in effect_name or
                any(effect_name in alias or alias in effect_name for alias in sound_effect.aliases)):
                return sound_effect.path
        
        return None
    
    def _suggest_category(self, effect_name: str) -> Optional[str]:
        """Suggest the most appropriate category for an effect name"""
        best_category = None
        best_score = 0
        
        for category, info in self.category_mapping.items():
            score = 0
            for keyword in info['keywords']:
                if keyword in effect_name:
                    score += 2  # Exact keyword match
                elif any(kw in effect_name for kw in keyword.split()):
                    score += 1  # Partial keyword match
            
            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category
    
    def _find_fallback_effect(self, effect_name: str) -> Optional[Path]:
        """Find a fallback effect using similarity and category fallbacks"""
        suggested_category = self._suggest_category(effect_name)
        
        if suggested_category and suggested_category in self.category_mapping:
            fallback_categories = self.category_mapping[suggested_category]['fallbacks']
            
            for fallback_category in fallback_categories:
                if fallback_category in self._sound_cache and self._sound_cache[fallback_category]:
                    # Return the first available effect in fallback category
                    return self._sound_cache[fallback_category][0].path
        
        # Last resort: return any available effect
        for category_effects in self._sound_cache.values():
            if category_effects:
                return category_effects[0].path
        
        return None
    
    def get_available_effects(self) -> Dict[str, List[str]]:
        """Get a summary of all available sound effects by category"""
        summary = {}
        for category, effects in self._sound_cache.items():
            summary[category] = [effect.name for effect in effects]
        return summary
    
    def validate_sound_effect(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Validate a sound effect file"""
        try:
            if not file_path.exists():
                return False, f"File does not exist: {file_path}"
            
            if file_path.suffix.lower() not in ['.wav', '.mp3', '.ogg', '.m4a']:
                return False, f"Unsupported audio format: {file_path.suffix}"
            
            # Try to load the audio file
            with AudioFileClip(str(file_path)) as audio_clip:
                if audio_clip.duration <= 0:
                    return False, "Audio file has zero duration"
                
                if audio_clip.duration > 10:  # Warn about long effects
                    return True, f"Warning: Audio file is quite long ({audio_clip.duration:.1f}s)"
            
            return True, None
            
        except Exception as e:
            return False, f"Error validating audio file: {e}"
    
    def analyze_sound_effect(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a sound effect file for metadata"""
        try:
            with AudioFileClip(str(file_path)) as audio_clip:
                # Get basic properties
                analysis = {
                    'duration': audio_clip.duration,
                    'fps': audio_clip.fps,
                    'file_size': file_path.stat().st_size,
                    'format': file_path.suffix.lower(),
                    'name': file_path.stem
                }
                
                # Estimate volume characteristics
                if audio_clip.duration > 0:
                    try:
                        # Sample a small portion to estimate volume
                        from src.processing.video_processor_fixes import MoviePyCompat
                        sample_audio = MoviePyCompat.subclip(audio_clip, 0, min(1.0, audio_clip.duration))
                        # This is a simplified analysis - could be enhanced with actual audio analysis
                        analysis['estimated_volume'] = 'medium'  # Placeholder
                    except Exception:
                        analysis['estimated_volume'] = 'unknown'
                
                return analysis
                
        except Exception as e:
            self.logger.warning(f"Error analyzing sound effect {file_path}: {e}")
            return None
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get current cache status and statistics"""
        total_effects = sum(len(effects) for effects in self._sound_cache.values())
        
        return {
            'total_effects': total_effects,
            'categories': len(self._sound_cache),
            'category_breakdown': {cat: len(effects) for cat, effects in self._sound_cache.items()},
            'sound_effects_dir': str(self.sound_effects_dir),
            'cache_populated': total_effects > 0
        }
    
    def apply_gen_z_sound_effects(self, analysis: Dict[str, Any], gen_z_mode: bool = False) -> List[Dict[str, Any]]:
        """
        Apply Gen Z sound effects based on content analysis
        
        Args:
            analysis: Video analysis data
            gen_z_mode: Whether Gen Z mode is enabled
            
        Returns:
            List of sound effect configurations to apply
        """
        if not gen_z_mode:
            return []
        
        try:
            sound_effects = []
            
            # Check if Gen Z mode is enabled in config
            config = get_config()
            if not config.ai_features.get('gen_z_mode', False):
                return []
            
            # Add Gen Z effects based on content analysis
            if analysis.get('humor_level', 0) > 0.6:
                # High humor content - add funny effects
                if analysis.get('key_moments'):
                    key_moment = analysis['key_moments'][0]
                    sound_effects.append({
                        'effect_name': 'vine_boom',
                        'timestamp_seconds': key_moment.get('timestamp', 0),
                        'volume': 0.8,
                        'category': 'gen_z',
                        'description': 'Vine boom for high humor moment'
                    })
            
            # Add effects for dramatic moments
            if analysis.get('dramatic_moments'):
                for moment in analysis['dramatic_moments'][:2]:  # Max 2 dramatic effects
                    if moment.get('intensity', 0) > 0.7:
                        sound_effects.append({
                            'effect_name': 'epic_sax',
                            'timestamp_seconds': moment.get('timestamp', 0),
                            'volume': 0.7,
                            'category': 'gen_z',
                            'description': 'Epic sax for dramatic moment'
                        })
            
            # Add effects for awkward/silent moments
            if analysis.get('awkward_moments'):
                for moment in analysis['awkward_moments'][:1]:  # Max 1 awkward effect
                    sound_effects.append({
                        'effect_name': 'cricket_chirp',
                        'timestamp_seconds': moment.get('timestamp', 0),
                        'volume': 0.6,
                        'category': 'gen_z',
                        'description': 'Cricket chirp for awkward moment'
                    })
            
            # Add effects for emotional moments
            if analysis.get('emotional_moments'):
                for moment in analysis['emotional_moments'][:1]:  # Max 1 emotional effect
                    if moment.get('emotion') == 'sad':
                        sound_effects.append({
                            'effect_name': 'sad_violin',
                            'timestamp_seconds': moment.get('timestamp', 0),
                            'volume': 0.5,
                            'category': 'gen_z',
                            'description': 'Sad violin for emotional moment'
                        })
            
            self.logger.info(f"🎵 Generated {len(sound_effects)} Gen Z sound effects")
            return sound_effects
            
        except Exception as e:
            self.logger.error(f"Failed to generate Gen Z sound effects: {e}")
            return []
    
    def find_gen_z_sound_effect(self, effect_name: str) -> Optional[Path]:
        """
        Find Gen Z specific sound effect
        
        Args:
            effect_name: Name of the Gen Z effect
            
        Returns:
            Path to sound effect file or None if not found
        """
        try:
            if effect_name not in self.gen_z_effects:
                return None
            
            effect_info = self.gen_z_effects[effect_name]
            
            # First try to find exact match
            exact_match = self.find_sound_effect(effect_name)
            if exact_match:
                return exact_match
            
            # Try fallback categories
            for fallback in effect_info['fallbacks']:
                fallback_match = self.find_sound_effect(effect_name, preferred_category=fallback)
                if fallback_match:
                    return fallback_match
            
            # Try keyword search
            for keyword in effect_info['keywords']:
                keyword_match = self.find_sound_effect(keyword)
                if keyword_match:
                    return keyword_match
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to find Gen Z sound effect '{effect_name}': {e}")
            return None