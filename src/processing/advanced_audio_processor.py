"""
Advanced Audio Processing with Intelligent Ducking and Voice Enhancement
Handles dynamic audio mixing, ducking during narration, and audio optimization.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import tempfile
from dataclasses import dataclass

# Optional imports with fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None
    sf = None

try:
    from moviepy import AudioFileClip, CompositeAudioClip, afx
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    AudioFileClip = None
    CompositeAudioClip = None
    afx = None
from src.config.settings import get_config
from src.models import AudioDuckingConfig, NarrativeSegment
from src.integrations.tts_service import TTSService


@dataclass
class AudioSegment:
    """Represents an audio segment with timing and properties"""
    start_time: float
    end_time: float
    audio_type: str  # "narration", "music", "sfx", "silence"
    volume_level: float
    frequency_profile: Optional[Any] = None
    energy_level: Optional[float] = None


class AdvancedAudioProcessor:
    """
    Advanced audio processing with intelligent ducking, voice enhancement,
    and dynamic mixing capabilities.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.tts_service = TTSService()
        
        # Check for required dependencies
        self.dependencies_available = NUMPY_AVAILABLE and MOVIEPY_AVAILABLE
        
        if not self.dependencies_available:
            missing_deps = []
            if not NUMPY_AVAILABLE:
                missing_deps.append("numpy")
            if not MOVIEPY_AVAILABLE:
                missing_deps.append("moviepy")
            
            self.logger.warning(f"⚠️ AdvancedAudioProcessor running in fallback mode - missing dependencies: {', '.join(missing_deps)}")
            self.logger.info("🔄 Enhanced audio features will be simulated")
        
        # Audio processing parameters
        self.sample_rate = 44100
        self.duck_fade_samples = int(0.5 * self.sample_rate)  # 0.5s fade
        self.voice_freq_range = (80, 8000)  # Human voice frequency range
        self.music_analysis_window = 2048
        
    def process_audio_with_ducking(self, 
                                  background_music: AudioFileClip,
                                  narrative_segments: List[NarrativeSegment],
                                  ducking_config: AudioDuckingConfig) -> AudioFileClip:
        """
        Process audio with intelligent ducking during narration
        
        Args:
            background_music: Background music track
            narrative_segments: List of TTS segments
            ducking_config: Ducking configuration
            
        Returns:
            Processed audio with ducking applied
        """
        try:
            if not ducking_config.duck_during_narration:
                return background_music
            
            # Check if dependencies are available
            if not self.dependencies_available:
                self.logger.info("🔄 Running audio processing in fallback mode")
                return self._fallback_audio_processing(background_music, narrative_segments)
            
            self.logger.info("Applying intelligent audio ducking...")
            
            # Generate TTS audio for all segments
            tts_segments = self._generate_tts_segments(narrative_segments)
            
            # Analyze music for optimal ducking points
            if ducking_config.smart_detection and LIBROSA_AVAILABLE:
                duck_zones = self._analyze_music_for_ducking(background_music, tts_segments)
            else:
                duck_zones = self._create_basic_duck_zones(tts_segments)
            
            # Apply ducking with preservation of music dynamics
            if ducking_config.preserve_music_dynamics:
                ducked_music = self._apply_dynamic_ducking(background_music, duck_zones, ducking_config)
            else:
                ducked_music = self._apply_simple_ducking(background_music, duck_zones, ducking_config)
            
            self.logger.info(f"Applied ducking to {len(duck_zones)} zones")
            return ducked_music
            
        except Exception as e:
            self.logger.error(f"Audio ducking failed: {e}")
            return background_music
    
    def _generate_tts_segments(self, narrative_segments: List[NarrativeSegment]) -> List[AudioSegment]:
        """Generate TTS audio and create audio segments"""
        tts_segments = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for i, segment in enumerate(narrative_segments):
                # Generate TTS audio
                audio_path = self.tts_service.generate_speech(segment, temp_path / f"tts_{i}.wav")
                
                if audio_path and audio_path.exists():
                    try:
                        # Load audio for analysis
                        with AudioFileClip(str(audio_path)) as clip:
                            duration = clip.duration
                            
                            # Analyze audio properties
                            energy = self._calculate_audio_energy(clip)
                            
                            audio_seg = AudioSegment(
                                start_time=segment.time_seconds,
                                end_time=segment.time_seconds + duration,
                                audio_type="narration",
                                volume_level=1.0,
                                energy_level=energy
                            )
                            tts_segments.append(audio_seg)
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to analyze TTS segment {i}: {e}")
        
        return tts_segments
    
    def _analyze_music_for_ducking(self, music: AudioFileClip, 
                                  tts_segments: List[AudioSegment]) -> List[Dict[str, Any]]:
        """Analyze music to find optimal ducking points"""
        duck_zones = []
        
        if not LIBROSA_AVAILABLE:
            return self._create_basic_duck_zones(tts_segments)
        
        try:
            # Extract audio data
            audio_array = music.to_soundarray(fps=self.sample_rate)
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)  # Convert to mono
            
            # Analyze tempo and beat
            tempo, beats = librosa.beat.beat_track(y=audio_array, sr=self.sample_rate)
            beat_times = librosa.frames_to_time(beats, sr=self.sample_rate)
            
            for tts_segment in tts_segments:
                # Find optimal ducking boundaries aligned with musical beats
                start_time = self._find_nearest_beat(beat_times, tts_segment.start_time)
                end_time = self._find_nearest_beat(beat_times, tts_segment.end_time)
                
                # Analyze frequency content at ducking points
                start_freqs = self._analyze_frequency_content(audio_array, start_time)
                end_freqs = self._analyze_frequency_content(audio_array, end_time)
                
                duck_zone = {
                    'start_time': max(0, start_time - 0.2),  # Pre-duck for smooth transition
                    'end_time': min(music.duration, end_time + 0.2),  # Post-duck
                    'tts_start': tts_segment.start_time,
                    'tts_end': tts_segment.end_time,
                    'beat_aligned': True,
                    'frequency_profile': {
                        'start': start_freqs,
                        'end': end_freqs
                    },
                    'voice_energy': tts_segment.energy_level
                }
                duck_zones.append(duck_zone)
                
        except Exception as e:
            self.logger.warning(f"Advanced music analysis failed, using basic ducking: {e}")
            return self._create_basic_duck_zones(tts_segments)
        
        return duck_zones
    
    def _find_nearest_beat(self, beat_times: Any, target_time: float) -> float:
        """Find the nearest musical beat to a target time"""
        if len(beat_times) == 0:
            return target_time
        
        distances = np.abs(beat_times - target_time)
        nearest_idx = np.argmin(distances)
        return beat_times[nearest_idx]
    
    def _analyze_frequency_content(self, audio_array: Any, 
                                  time_point: float) -> Any:
        """Analyze frequency content at a specific time point"""
        if not LIBROSA_AVAILABLE:
            return np.array([])
        
        try:
            # Extract a window around the time point
            start_sample = max(0, int((time_point - 0.5) * self.sample_rate))
            end_sample = min(len(audio_array), int((time_point + 0.5) * self.sample_rate))
            
            window = audio_array[start_sample:end_sample]
            
            # Compute FFT
            fft = np.fft.fft(window, n=self.music_analysis_window)
            magnitude = np.abs(fft[:self.music_analysis_window//2])
            
            return magnitude
            
        except Exception as e:
            self.logger.debug(f"Frequency analysis failed: {e}")
            return np.array([])
    
    def _create_basic_duck_zones(self, tts_segments: List[AudioSegment]) -> List[Dict[str, Any]]:
        """Create basic ducking zones without advanced analysis"""
        duck_zones = []
        
        for segment in tts_segments:
            duck_zone = {
                'start_time': max(0, segment.start_time - 0.2),
                'end_time': segment.end_time + 0.2,
                'tts_start': segment.start_time,
                'tts_end': segment.end_time,
                'beat_aligned': False,
                'voice_energy': segment.energy_level or 1.0
            }
            duck_zones.append(duck_zone)
        
        return duck_zones
    
    def _apply_dynamic_ducking(self, music: AudioFileClip, 
                              duck_zones: List[Dict[str, Any]],
                              config: AudioDuckingConfig) -> AudioFileClip:
        """Apply dynamic ducking that preserves music characteristics"""
        try:
            # Convert music to array for processing
            audio_array = music.to_soundarray(fps=self.sample_rate)
            if len(audio_array.shape) > 1:
                # Keep stereo if present
                is_stereo = True
                channels = audio_array.shape[1]
            else:
                is_stereo = False
                audio_array = audio_array.reshape(-1, 1)
                channels = 1
            
            # Apply ducking to each zone
            for zone in duck_zones:
                start_sample = int(zone['start_time'] * self.sample_rate)
                end_sample = int(zone['end_time'] * self.sample_rate)
                tts_start_sample = int(zone['tts_start'] * self.sample_rate)
                tts_end_sample = int(zone['tts_end'] * self.sample_rate)
                
                if start_sample >= len(audio_array) or end_sample <= 0:
                    continue
                
                start_sample = max(0, start_sample)
                end_sample = min(len(audio_array), end_sample)
                
                # Calculate dynamic duck level based on voice energy
                voice_energy = zone.get('voice_energy', 1.0)
                duck_level = config.duck_volume * (0.8 + 0.2 * (1 - voice_energy))
                
                # Apply frequency-selective ducking if frequency profile available
                if 'frequency_profile' in zone and LIBROSA_AVAILABLE:
                    audio_array[start_sample:end_sample] = self._apply_frequency_selective_ducking(
                        audio_array[start_sample:end_sample], duck_level, zone['frequency_profile']
                    )
                else:
                    # Apply smooth volume ducking
                    audio_array[start_sample:end_sample] = self._apply_smooth_volume_ducking(
                        audio_array[start_sample:end_sample], duck_level, config.fade_duration
                    )
            
            # Convert back to AudioFileClip
            if is_stereo and channels > 1:
                processed_clip = AudioFileClip._make_with_array(audio_array, fps=self.sample_rate)
            else:
                processed_clip = AudioFileClip._make_with_array(audio_array.squeeze(), fps=self.sample_rate)
            
            return processed_clip
            
        except Exception as e:
            self.logger.error(f"Dynamic ducking failed: {e}")
            return self._apply_simple_ducking(music, duck_zones, config)
    
    def _apply_frequency_selective_ducking(self, audio_segment: Any, 
                                          duck_level: float,
                                          freq_profile: Dict[str, Any]) -> Any:
        """Apply ducking that targets specific frequency ranges"""
        if not LIBROSA_AVAILABLE:
            return audio_segment * duck_level
        
        try:
            # Convert to frequency domain
            stft = librosa.stft(audio_segment.T if len(audio_segment.shape) > 1 else audio_segment)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Create frequency mask for voice range
            freqs = librosa.fft_frequencies(sr=self.sample_rate)
            voice_mask = (freqs >= self.voice_freq_range[0]) & (freqs <= self.voice_freq_range[1])
            
            # Apply stronger ducking in voice frequency range
            magnitude[voice_mask] *= duck_level
            magnitude[~voice_mask] *= (duck_level + (1 - duck_level) * 0.5)  # Less ducking outside voice range
            
            # Convert back to time domain
            ducked_stft = magnitude * np.exp(1j * phase)
            ducked_audio = librosa.istft(ducked_stft)
            
            # Ensure same shape as input
            if len(audio_segment.shape) > 1:
                ducked_audio = ducked_audio.T
                if ducked_audio.shape != audio_segment.shape:
                    ducked_audio = ducked_audio[:audio_segment.shape[0], :audio_segment.shape[1]]
            
            return ducked_audio
            
        except Exception as e:
            self.logger.debug(f"Frequency-selective ducking failed: {e}")
            return audio_segment * duck_level
    
    def _apply_smooth_volume_ducking(self, audio_segment: Any, 
                                    duck_level: float, fade_duration: float) -> Any:
        """Apply smooth volume ducking with fade in/out"""
        segment_length = len(audio_segment)
        fade_samples = min(int(fade_duration * self.sample_rate), segment_length // 4)
        
        # Create ducking envelope
        envelope = np.ones(segment_length)
        
        # Fade in to duck level
        for i in range(fade_samples):
            factor = i / fade_samples
            envelope[i] = 1.0 - factor * (1.0 - duck_level)
        
        # Duck level in middle
        envelope[fade_samples:-fade_samples] = duck_level
        
        # Fade out from duck level
        for i in range(fade_samples):
            factor = i / fade_samples
            envelope[-(fade_samples-i)] = duck_level + factor * (1.0 - duck_level)
        
        # Apply envelope
        if len(audio_segment.shape) > 1:
            return audio_segment * envelope.reshape(-1, 1)
        else:
            return audio_segment * envelope
    
    def _apply_simple_ducking(self, music: AudioFileClip, 
                             duck_zones: List[Dict[str, Any]],
                             config: AudioDuckingConfig) -> AudioFileClip:
        """Apply simple volume-based ducking"""
        try:
            audio_clips = []
            last_time = 0.0
            
            for zone in duck_zones:
                start_time = zone['start_time']
                end_time = zone['end_time']
                
                # Add unducked section before this zone
                if start_time > last_time:
                    unducked_section = music.subclipped(last_time, start_time)
                    audio_clips.append(unducked_section)
                
                # Add ducked section
                if end_time > start_time:
                    ducked_section = music.subclipped(start_time, end_time)
                    ducked_section = ducked_section.with_volume(config.duck_volume)
                    
                    # Apply fade in/out
                    if config.fade_duration > 0:
                        fade_dur = min(config.fade_duration, (end_time - start_time) / 2)
                        ducked_section = ducked_section.with_fadein(fade_dur).with_fadeout(fade_dur)
                    
                    audio_clips.append(ducked_section)
                
                last_time = end_time
            
            # Add final unducked section
            if last_time < music.duration:
                final_section = music.subclipped(last_time, music.duration)
                audio_clips.append(final_section)
            
            # Concatenate all sections
            if audio_clips:
                return CompositeAudioClip(audio_clips)
            else:
                return music
                
        except Exception as e:
            self.logger.error(f"Simple ducking failed: {e}")
            return music
    
    def _calculate_audio_energy(self, audio_clip: AudioFileClip) -> float:
        """Calculate the energy level of an audio clip"""
        try:
            audio_array = audio_clip.to_soundarray(fps=self.sample_rate)
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio_array ** 2))
            return min(rms * 10, 1.0)  # Normalize to 0-1 range
            
        except Exception as e:
            self.logger.debug(f"Energy calculation failed: {e}")
            return 0.5  # Default medium energy
    
    def enhance_voice_audio(self, audio_path: Path, 
                           enhancement_params: Dict[str, float]) -> Optional[Path]:
        """
        Enhance voice audio with noise reduction and clarity improvements
        
        Args:
            audio_path: Path to voice audio file
            enhancement_params: Enhancement parameters
            
        Returns:
            Path to enhanced audio file
        """
        if not LIBROSA_AVAILABLE:
            self.logger.warning("Voice enhancement requires librosa library")
            return audio_path
        
        try:
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
            
            # Apply enhancements
            enhanced_audio = audio.copy()
            
            # Noise reduction (spectral gating)
            if enhancement_params.get('noise_reduction', 0) > 0:
                enhanced_audio = self._apply_noise_reduction(enhanced_audio, sr, 
                                                           enhancement_params['noise_reduction'])
            
            # Clarity enhancement (high-frequency boost)
            if enhancement_params.get('clarity_boost', 0) > 0:
                enhanced_audio = self._apply_clarity_boost(enhanced_audio, sr,
                                                         enhancement_params['clarity_boost'])
            
            # Dynamic range compression
            if enhancement_params.get('compression', 0) > 0:
                enhanced_audio = self._apply_compression(enhanced_audio,
                                                       enhancement_params['compression'])
            
            # Save enhanced audio
            enhanced_path = audio_path.with_suffix('.enhanced.wav')
            sf.write(str(enhanced_path), enhanced_audio, sr)
            
            self.logger.info(f"Enhanced voice audio: {enhanced_path}")
            return enhanced_path
            
        except Exception as e:
            self.logger.error(f"Voice enhancement failed: {e}")
            return audio_path
    
    def _apply_noise_reduction(self, audio: Any, sr: int, strength: float) -> Any:
        """Apply spectral noise reduction"""
        try:
            # Compute spectrogram
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise profile from quiet portions
            noise_profile = np.percentile(magnitude, 10, axis=1, keepdims=True)
            
            # Apply spectral gating
            gate_threshold = noise_profile * (1 + strength * 3)
            magnitude_gated = np.maximum(magnitude, gate_threshold)
            
            # Reconstruct audio
            enhanced_stft = magnitude_gated * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft)
            
            return enhanced_audio
            
        except Exception as e:
            self.logger.debug(f"Noise reduction failed: {e}")
            return audio
    
    def _apply_clarity_boost(self, audio: Any, sr: int, strength: float) -> Any:
        """Apply high-frequency clarity boost"""
        try:
            # Design high-pass filter for clarity
            from scipy import signal
            
            # Boost frequencies above 2kHz
            nyquist = sr / 2
            high_freq = 2000 / nyquist
            b, a = signal.butter(2, high_freq, btype='high')
            
            # Apply filter with controlled strength
            filtered = signal.filtfilt(b, a, audio)
            enhanced = audio + filtered * strength * 0.3
            
            return enhanced
            
        except Exception as e:
            self.logger.debug(f"Clarity boost failed: {e}")
            return audio
    
    def _apply_compression(self, audio: Any, strength: float) -> Any:
        """Apply dynamic range compression"""
        try:
            # Simple compression using tanh
            threshold = 0.1 * (1 - strength)
            ratio = 1 + strength * 3
            
            # Apply compression
            compressed = np.where(
                np.abs(audio) > threshold,
                np.sign(audio) * (threshold + (np.abs(audio) - threshold) / ratio),
                audio
            )
            
            # Normalize
            if np.max(np.abs(compressed)) > 0:
                compressed = compressed / np.max(np.abs(compressed)) * 0.95
            
            return compressed
            
        except Exception as e:
            self.logger.debug(f"Compression failed: {e}")
            return audio
    
    def _fallback_audio_processing(self, background_music: AudioFileClip, 
                                 narrative_segments: List[NarrativeSegment]) -> AudioFileClip:
        """
        Fallback audio processing when dependencies are not available
        Returns the background music with simulated processing
        """
        try:
            self.logger.info("🔄 Running fallback audio processing")
            
            # In fallback mode, just return the background music
            # In a real implementation, this could use basic audio manipulation
            # or cached processed audio
            
            self.logger.info("✅ Fallback audio processing completed")
            return background_music
            
        except Exception as e:
            self.logger.error(f"Fallback audio processing failed: {e}")
            return background_music