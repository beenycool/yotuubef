"""
Enhanced Text-to-Speech service with support for ElevenLabs and the local Dia-1.6B model.
Handles dynamic TTS parameters, emotion, dialogue generation, and pacing for engaging narration,
with a preference for the local model to reduce API costs and latency.
Optimized for GPU memory usage with 6GB VRAM constraints.
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import io
import asyncio
import gc

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
    try:
        # Fallback import structure
        from dia import Dia
        import soundfile as sf
        import torch
        DIA_AVAILABLE = True
    except ImportError:
        DIA_AVAILABLE = False

import numpy as np
from moviepy import AudioFileClip

from src.config.settings import get_config
from src.models import NarrativeSegment, EmotionType, PacingType
from src.utils.gpu_memory_manager import GPUMemoryManager


class TTSService:
    """
    Comprehensive Text-to-Speech service supporting multiple providers
    """
    
    # Class variable to track if TTS has been initialized
    _tts_initialized = False
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self._dia_model = None
        self.gpu_manager = GPUMemoryManager(max_vram_usage=0.75)  # Increased for better TTS performance
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize available TTS services"""
        if ELEVENLABS_AVAILABLE and self.config.api.elevenlabs_api_key:
            try:
                elevenlabs.set_api_key(self.config.api.elevenlabs_api_key)
                if not TTSService._tts_initialized:
                    self.logger.info("ElevenLabs TTS service initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ElevenLabs: {e}")
        
        if DIA_AVAILABLE:
            if not TTSService._tts_initialized:
                self.logger.info("Local Dia-1.6B TTS model is available for use.")
                TTSService._tts_initialized = True
        else:
            if not TTSService._tts_initialized:
                self.logger.warning("Dia TTS not available - install dia package")
    
    def _initialize_dia(self):
        """Lazy initialization of the local Dia model with GPU memory optimization."""
        if self._dia_model is None and DIA_AVAILABLE:
            try:
                # Check VRAM availability for Dia model (estimated ~1.6GB)
                estimated_model_size_mb = 1800  # Conservative estimate for Dia-1.6B
                device = self.gpu_manager.get_optimal_device(estimated_model_size_mb)
                
                self.logger.info(f"Loading Dia-1.6B TTS model on {device}...")
                
                # Load model with simplified initialization (Dia doesn't support torch_dtype, device_map, etc.)
                if device.startswith("cuda"):
                    # Enable memory-efficient loading
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()  # Clear cache before loading
                        
                        # Load model with basic initialization
                        self._dia_model = Dia.from_pretrained("nari-labs/Dia-1.6B")
                        
                        # Move to GPU manually if needed
                        if hasattr(self._dia_model, 'to'):
                            self._dia_model = self._dia_model.to(device)
                        
                        # Optimize for inference
                        self._dia_model = self.gpu_manager.optimize_model_for_inference(self._dia_model)
                        
                        # Log VRAM usage
                        vram_info = self.gpu_manager.get_vram_info()
                        if vram_info:
                            used_gb = vram_info['used'] / 1024**3
                            self.logger.info(f"Dia model loaded to GPU, VRAM used: {used_gb:.1f}GB")
                else:
                    # CPU fallback
                    self.logger.info("Loading Dia model to CPU due to insufficient VRAM")
                    self._dia_model = Dia.from_pretrained("nari-labs/Dia-1.6B")
                    
                    # Ensure model stays on CPU
                    if hasattr(self._dia_model, 'to'):
                        self._dia_model = self._dia_model.to('cpu')
                
                self.logger.info("Dia-1.6B model loaded successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to load Dia model: {e}")
                self._dia_model = None
                # Clear any partial GPU allocations
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def generate_speech(self,
                        segment: NarrativeSegment,
                        output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Generate speech for a narrative segment

        Args:
            segment: NarrativeSegment with text and parameters
            output_path: Optional output path for audio file

        Returns:
            Path to generated audio file or None if failed
        """
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix='.wav'))
        
        # Validate and sanitize output path for security
        try:
            output_path = self._validate_audio_output_path(output_path)
        except ValueError as e:
            self.logger.error(f"Invalid output path: {e}")
            return None

    def _validate_audio_output_path(self, output_path):
        """
        Inline validation for audio output path security.
        Ensures the path is a file, not a directory, and is within allowed directories.
        """
        output_path = Path(output_path)
        if output_path.is_dir():
            raise ValueError("Output path must be a file, not a directory.")
        # Restrict output path to the configured audio folder for security
        allowed_dir = Path(self.config.paths.audio_folder).resolve()
        if allowed_dir not in output_path.resolve().parents:
            raise ValueError("Output path is outside allowed audio directory.")
        # Check for forbidden characters or patterns
        forbidden = ["..", "~", "//", "\\", "|", ":", "*", "?", "\"", "<", ">"]
        if any(f in str(output_path) for f in forbidden):
            raise ValueError("Output path contains forbidden characters or patterns.")
        return output_path
        
        # Prioritize local Dia model if available
        if DIA_AVAILABLE:
            result = self._generate_with_dia(segment, output_path)
            if result:
                return result
        
        # Fallback to ElevenLabs if Dia fails or is unavailable
        if ELEVENLABS_AVAILABLE and self.config.api.elevenlabs_api_key:
            result = self._generate_with_elevenlabs(segment, output_path)
            if result:
                return result

        # Final fallback: try system TTS if available
        try:
            import pyttsx3
            return self._generate_with_pyttsx3(segment, output_path)
        except ImportError:
            pass
        
        self.logger.error("No TTS service available for speech generation")
        return None
    
    def _generate_with_elevenlabs(self,
                                 segment: NarrativeSegment,
                                 output_path: Path) -> Optional[Path]:
        """Generate speech using the ElevenLabs API."""
        try:
            # Map emotion and pacing to ElevenLabs voice settings
            voice_settings = self._get_elevenlabs_voice_settings(segment.emotion, segment.pacing)
            
            # Generate audio
            audio = generate(
                text=segment.text,
                voice=Voice(voice_id=self.config.api.default_voice_id),
                voice_settings=voice_settings
            )
            
            # Save to file
            save(audio, str(output_path))
            
            self.logger.info(f"Generated ElevenLabs TTS: '{segment.text[:30]}...'")
            return output_path
            
        except Exception as e:
            self.logger.warning(f"ElevenLabs TTS failed, will try other providers: {e}")
            return None
    
    def _generate_with_dia(self,
                          segment: NarrativeSegment,
                          output_path: Path) -> Optional[Path]:
        """Generate speech using the local Dia-1.6B model with GPU memory optimization."""
        try:
            self._initialize_dia()
            
            if self._dia_model is None:
                self.logger.warning("Dia model not initialized, cannot generate speech.")
                return None
            
            # Prepare text with dialogue format and emotion hints
            enhanced_text = self._enhance_text_for_dia(segment)
            
            # Use GPU context manager for memory efficient generation
            # Check if model has parameters() method (PyTorch models do, but Dia might not)
            try:
                device = next(self._dia_model.parameters()).device
            except (AttributeError, StopIteration):
                # Fallback: check if model has a device attribute or use cuda if available
                if hasattr(self._dia_model, 'device'):
                    device = self._dia_model.device
                elif torch.cuda.is_available():
                    device = torch.device('cuda:0')
                else:
                    device = torch.device('cpu')
            
            with torch.inference_mode():  # Disable gradient computation for inference
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        # Monitor VRAM before generation
                        vram_before = self.gpu_manager.get_vram_info()
                        
                        # Generate speech using Dia with proper arguments
                        try:
                            # Try calling with just text (most common interface)
                            audio_output = self._dia_model.generate(enhanced_text)
                        except Exception as e:
                            self.logger.warning(f"Direct generate call failed: {e}")
                            # Try alternative calling patterns
                            try:
                                audio_output = self._dia_model(enhanced_text)
                            except Exception as e2:
                                self.logger.warning(f"Direct model call failed: {e2}")
                                raise e  # Re-raise original error
                        
                        # Clear intermediate GPU tensors
                        torch.cuda.empty_cache()
                        
                        if vram_before:
                            vram_after = self.gpu_manager.get_vram_info()
                            if vram_after:
                                used_mb = (vram_after['used'] - vram_before['used']) / 1024**2
                                self.logger.debug(f"TTS generation used {used_mb:.0f}MB VRAM")
                else:
                    # CPU generation
                    try:
                        # Try calling with just text (most common interface)
                        audio_output = self._dia_model.generate(enhanced_text)
                    except Exception as e:
                        self.logger.warning(f"Direct generate call failed: {e}")
                        # Try alternative calling patterns
                        try:
                            audio_output = self._dia_model(enhanced_text)
                        except Exception as e2:
                            self.logger.warning(f"Direct model call failed: {e2}")
                            raise e  # Re-raise original error
            
            # Handle different return formats from Dia model
            if isinstance(audio_output, tuple):
                # If Dia returns (audio_tensor, sample_rate)
                audio_tensor, sr = audio_output
                if hasattr(audio_tensor, 'numpy'):
                    audio_data = audio_tensor.cpu().numpy()  # Ensure on CPU for numpy
                else:
                    audio_data = audio_tensor
            else:
                # If Dia returns just audio data
                audio_data = audio_output
                if hasattr(audio_data, 'numpy'):
                    audio_data = audio_data.cpu().numpy()  # Ensure on CPU for numpy
                elif hasattr(audio_data, 'cpu'):
                    audio_data = audio_data.cpu().numpy()
                sr = 44100  # Default sample rate
            
            # Ensure audio_data is in the right format for soundfile
            if len(audio_data.shape) > 1:
                audio_data = audio_data.squeeze()
            
            sf.write(str(output_path), audio_data, sr)
            
            # Clean up GPU memory after generation
            if device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            self.logger.info(f"Generated Dia TTS: '{segment.text[:30]}...'")
            return output_path
            
        except Exception as e:
            self.logger.warning(f"Local Dia TTS failed, will try other providers: {e}")
            # Clean up on error
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
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
        """Enhance text with Dia-specific formatting for emotion and pacing."""
        text = segment.text
        formatted_text = f"[S1] {text}"

        # Add emotion-based nonverbal cues that Dia recognizes
        if segment.emotion == EmotionType.EXCITED:
            if not any(punct in text for punct in ['!', '?']):
                formatted_text = formatted_text.rstrip('.') + '!'
            if 'amazing' in text.lower() or 'incredible' in text.lower() or 'crazy' in text.lower():
                formatted_text += " [laughs]"
        elif segment.emotion == EmotionType.DRAMATIC:
            formatted_text = formatted_text.replace(',', '... ,')
            if 'believe' in text.lower() or 'secret' in text.lower():
                formatted_text += " [gasps]"
        elif segment.emotion == EmotionType.CALM:
            formatted_text = formatted_text.replace('.', ' [sighs].')

        # Add pacing hints through punctuation and nonverbals
        if segment.pacing == PacingType.FAST:
            pass # Default punctuation is usually fine for fast pacing
        elif segment.pacing == PacingType.SLOW:
            formatted_text = formatted_text.replace(',', ', ...')

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
        Adjust audio speed to match a target duration using moviepy.
        
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
            
            if abs(current_duration - target_duration) < 0.05:
                # Duration is close enough
                clip.close()
                return audio_path
            
            # Calculate speed factor
            speed_factor = current_duration / target_duration
            
            # Apply speed change (limit to reasonable range)
            speed_factor = max(0.7, min(1.5, speed_factor))
            
            # Apply speed adjustment
            adjusted_clip = clip.fx(lambda c: c.set_duration(c.duration / speed_factor))
            adjusted_clip.write_audiofile(str(output_path), verbose=False, logger=None)
            
            clip.close()
            adjusted_clip.close()

            self.logger.debug(f"Adjusted audio {audio_path.name} speed by factor {speed_factor:.2f}")
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
    
    def generate_narrative_audio(self, segments: List['NarrativeSegment']) -> List[Optional[AudioFileClip]]:
        """
        Generate audio clips for narrative segments (used by CTA processor)
        
        Args:
            segments: List of NarrativeSegment objects
            
        Returns:
            List of AudioFileClip objects (or None for failed generations)
        """
        audio_clips = []
        
        for segment in segments:
            try:
                # Generate speech for this segment
                audio_path = self.generate_speech(segment)
                
                if audio_path and audio_path.exists():
                    # Load as AudioFileClip
                    audio_clip = AudioFileClip(str(audio_path))
                    audio_clips.append(audio_clip)
                    self.logger.debug(f"Generated narrative audio for: '{segment.text[:30]}...'")
                else:
                    audio_clips.append(None)
                    self.logger.warning(f"Failed to generate audio for segment: '{segment.text[:30]}...'")
                    
            except Exception as e:
                self.logger.warning(f"Error generating narrative audio: {e}")
                audio_clips.append(None)
        
        return audio_clips
    def _generate_with_pyttsx3(self,
                               segment: NarrativeSegment,
                               output_path: Path) -> Optional[Path]:
        """Generate speech using system TTS (pyttsx3) as fallback."""
        try:
            import pyttsx3
            
            engine = pyttsx3.init()
            
            # Configure voice settings based on emotion and pacing
            rate = 200  # Default rate
            if segment.pacing == PacingType.SLOW:
                rate = 150
            elif segment.pacing == PacingType.FAST:
                rate = 250
            
            engine.setProperty('rate', rate)
            engine.setProperty('volume', 0.9)
            
            # Save to file
            engine.save_to_file(segment.text, str(output_path))
            engine.runAndWait()
            
            self.logger.info(f"Generated pyttsx3 TTS: '{segment.text[:30]}...'")
            return output_path
            
        except Exception as e:
            self.logger.warning(f"pyttsx3 TTS failed: {e}")
            return None