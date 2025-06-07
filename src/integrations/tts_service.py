"""
Enhanced Text-to-Speech service with support for ElevenLabs and Nari Labs Dia-1.6B.
Handles dynamic TTS parameters, emotion, dialogue generation, and pacing for engaging narration.
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import io
import asyncio

try:
    import elevenlabs
    from elevenlabs import Voice, VoiceSettings, generate, save
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False

try:
    import torch
    import soundfile as sf
    from dia.model import Dia
    DIA_AVAILABLE = True
except ImportError:
    DIA_AVAILABLE = False

import numpy as np
from moviepy.editor import AudioFileClip

from src.config.settings import get_config
from src.models import NarrativeSegment, EmotionType, PacingType


class TTSService:
    """
    Comprehensive Text-to-Speech service supporting multiple providers
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self._dia_model = None
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize available TTS services"""
        # Initialize ElevenLabs if available and configured
        if ELEVENLABS_AVAILABLE and hasattr(self.config, 'elevenlabs_api_key'):
            try:
                elevenlabs.set_api_key(self.config.elevenlabs_api_key)
                self.logger.info("ElevenLabs TTS service initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ElevenLabs: {e}")
        
        # Check Dia availability
        if DIA_AVAILABLE:
            self.logger.info("Dia-1.6B TTS model available for initialization")
        else:
            self.logger.warning("Dia TTS not available - install dia package")
    
    def _initialize_dia(self):
        """Lazy initialization of Dia model"""
        if self._dia_model is None and DIA_AVAILABLE:
            try:
                self.logger.info("Loading Dia-1.6B TTS model...")
                self._dia_model = Dia.from_pretrained("nari-labs/Dia-1.6B")
                self.logger.info("Dia-1.6B model loaded successfully")
                    
            except Exception as e:
                self.logger.error(f"Failed to load Dia model: {e}")
                self._dia_model = None
    
    def generate_speech(self, 
                       segment: NarrativeSegment, 
                       output_path: Optional[Path] = None,
                       prefer_elevenlabs: bool = True) -> Optional[Path]:
        """
        Generate speech for a narrative segment
        
        Args:
            segment: NarrativeSegment with text and parameters
            output_path: Optional output path for audio file
            prefer_elevenlabs: Whether to prefer ElevenLabs over Bark
        
        Returns:
            Path to generated audio file or None if failed
        """
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix='.wav'))
        
        # Try preferred service first
        if prefer_elevenlabs and ELEVENLABS_AVAILABLE:
            result = self._generate_with_elevenlabs(segment, output_path)
            if result:
                return result
        
        # Fallback to Dia
        if DIA_AVAILABLE:
            result = self._generate_with_dia(segment, output_path)
            if result:
                return result
        
        # Final fallback to ElevenLabs if not tried yet
        if not prefer_elevenlabs and ELEVENLABS_AVAILABLE:
            result = self._generate_with_elevenlabs(segment, output_path)
            if result:
                return result
        
        self.logger.error("No TTS service available for speech generation")
        return None
    
    def _generate_with_elevenlabs(self, 
                                 segment: NarrativeSegment, 
                                 output_path: Path) -> Optional[Path]:
        """Generate speech using ElevenLabs API"""
        try:
            # Map emotion and pacing to ElevenLabs voice settings
            voice_settings = self._get_elevenlabs_voice_settings(segment.emotion, segment.pacing)
            
            # Generate audio
            audio = generate(
                text=segment.text,
                voice=self.config.apis.elevenlabs.default_voice,
                voice_settings=voice_settings
            )
            
            # Save to file
            save(audio, str(output_path))
            
            self.logger.info(f"Generated ElevenLabs TTS for: '{segment.text[:50]}...'")
            return output_path
            
        except Exception as e:
            self.logger.warning(f"ElevenLabs TTS failed: {e}")
            return None
    
    def _generate_with_dia(self,
                          segment: NarrativeSegment,
                          output_path: Path) -> Optional[Path]:
        """Generate speech using Dia-1.6B model"""
        try:
            # Initialize Dia if needed
            self._initialize_dia()
            
            if self._dia_model is None:
                return None
            
            # Prepare text with dialogue format and emotion hints
            enhanced_text = self._enhance_text_for_dia(segment)
            
            # Generate speech using Dia
            output = self._dia_model.generate(enhanced_text)
            
            # Save as WAV file (Dia outputs at 44.1kHz)
            sf.write(str(output_path), output, 44100)
            
            self.logger.info(f"Generated Dia TTS for: '{segment.text[:50]}...'")
            return output_path
            
        except Exception as e:
            self.logger.warning(f"Dia TTS failed: {e}")
            return None
    
    def _get_elevenlabs_voice_settings(self, 
                                     emotion: EmotionType, 
                                     pacing: PacingType) -> VoiceSettings:
        """Map emotion and pacing to ElevenLabs voice settings"""
        # Base settings
        stability = 0.5
        similarity_boost = 0.75
        style = 0.5
        use_speaker_boost = True
        
        # Adjust based on emotion
        if emotion == EmotionType.EXCITED:
            stability = 0.3  # More variation
            style = 0.8      # More expressive
        elif emotion == EmotionType.DRAMATIC:
            stability = 0.4
            style = 0.9      # Very expressive
        elif emotion == EmotionType.CALM:
            stability = 0.8  # More stable
            style = 0.3      # Less expressive
        
        # Note: ElevenLabs doesn't directly control pacing via settings
        # Pacing would be controlled through text modifications or post-processing
        
        return VoiceSettings(
            stability=stability,
            similarity_boost=similarity_boost,
            style=style,
            use_speaker_boost=use_speaker_boost
        )
    
    def _enhance_text_for_dia(self, segment: NarrativeSegment) -> str:
        """Enhance text with Dia-specific formatting for emotion, pacing, and dialogue"""
        text = segment.text
        
        # Dia uses speaker tags and nonverbal cues for more realistic dialogue
        # Format text with speaker tags for better voice consistency
        formatted_text = f"[S1] {text}"
        
        # Add emotion-based nonverbal cues that Dia recognizes
        if segment.emotion == EmotionType.EXCITED:
            # Add enthusiasm markers
            if not any(punct in text for punct in ['!', '?']):
                formatted_text = formatted_text.rstrip('.') + '!'
            # Dia can generate excited laughter
            if 'amazing' in text.lower() or 'incredible' in text.lower():
                formatted_text += " (laughs)"
                
        elif segment.emotion == EmotionType.DRAMATIC:
            # Add dramatic pauses and gasps
            formatted_text = formatted_text.replace(',', ' (pause),')
            if 'believe' in text.lower() or 'wow' in text.lower():
                formatted_text += " (gasps)"
                
        elif segment.emotion == EmotionType.CALM:
            # Add gentle breathing and softer delivery
            formatted_text = formatted_text.replace('.', ' (sighs).')
        
        # Add pacing hints through punctuation and nonverbals
        if segment.pacing == PacingType.FAST:
            # Quick delivery with shorter pauses
            formatted_text = formatted_text.replace(',', ',')
            formatted_text = formatted_text.replace('!', '!')
            
        elif segment.pacing == PacingType.SLOW:
            # Slow delivery with longer pauses and breathing
            formatted_text = formatted_text.replace(',', '... ')
            formatted_text = formatted_text.replace('.', '... (inhales)')
        
        # Ensure proper dialogue format for Dia
        return formatted_text
    
    def generate_multiple_segments(self, 
                                 segments: List[NarrativeSegment],
                                 output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Generate speech for multiple narrative segments
        
        Args:
            segments: List of NarrativeSegment objects
            output_dir: Directory to save audio files
        
        Returns:
            List of dictionaries with segment info and audio paths
        """
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix='tts_'))
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, segment in enumerate(segments):
            try:
                output_path = output_dir / f"segment_{i:03d}.wav"
                audio_path = self.generate_speech(segment, output_path)
                
                result = {
                    'segment': segment,
                    'audio_path': audio_path,
                    'index': i,
                    'success': audio_path is not None
                }
                
                if audio_path:
                    # Get audio duration for timing validation
                    try:
                        clip = AudioFileClip(str(audio_path))
                        result['actual_duration'] = clip.duration
                        clip.close()
                    except Exception as e:
                        self.logger.warning(f"Could not get duration for segment {i}: {e}")
                        result['actual_duration'] = segment.intended_duration_seconds
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to generate TTS for segment {i}: {e}")
                results.append({
                    'segment': segment,
                    'audio_path': None,
                    'index': i,
                    'success': False,
                    'error': str(e)
                })
        
        successful = sum(1 for r in results if r['success'])
        self.logger.info(f"Generated TTS for {successful}/{len(segments)} segments")
        
        return results
    
    def adjust_audio_speed(self, 
                          audio_path: Path, 
                          target_duration: float,
                          output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Adjust audio speed to match target duration
        
        Args:
            audio_path: Path to source audio file
            target_duration: Target duration in seconds
            output_path: Optional output path
        
        Returns:
            Path to speed-adjusted audio file
        """
        if output_path is None:
            output_path = audio_path.with_suffix('.adjusted.wav')
        
        try:
            clip = AudioFileClip(str(audio_path))
            current_duration = clip.duration
            
            if abs(current_duration - target_duration) < 0.1:
                # Duration is close enough
                clip.close()
                return audio_path
            
            # Calculate speed factor
            speed_factor = current_duration / target_duration
            
            # Apply speed change (limit to reasonable range)
            speed_factor = max(0.5, min(2.0, speed_factor))
            
            # Apply speed adjustment
            adjusted_clip = clip.fx(lambda clip: clip.set_duration(target_duration))
            adjusted_clip.write_audiofile(str(output_path), verbose=False, logger=None)
            
            clip.close()
            adjusted_clip.close()
            
            self.logger.debug(f"Adjusted audio speed by factor {speed_factor:.2f}")
            return output_path
            
        except Exception as e:
            self.logger.warning(f"Failed to adjust audio speed: {e}")
            return audio_path
    
    def is_available(self) -> bool:
        """Check if any TTS service is available"""
        return ELEVENLABS_AVAILABLE or DIA_AVAILABLE
    
    def get_available_services(self) -> List[str]:
        """Get list of available TTS services"""
        services = []
        if ELEVENLABS_AVAILABLE:
            services.append("elevenlabs")
        if DIA_AVAILABLE:
            services.append("dia")
        return services