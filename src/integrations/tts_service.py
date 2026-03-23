"""
Text-to-Speech service with support for local Qwen3-TTS 1.7B CustomVoice.
Handles dynamic TTS parameters, emotion, and pacing for engaging narration.
Optimized for GPU memory usage with constrained VRAM environments.
"""

import logging
import importlib
import importlib.util
from moviepy import AudioFileClip

from src.config.settings import get_config
from src.models import NarrativeSegment, EmotionType, PacingType
from src.utils.gpu_memory_manager import GPUMemoryManager
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import gc
import numpy as np

_torch: Any = None
_sf: Any = None
_librosa: Any = None
_Qwen3TTSModel: Any = None

try:
    _torch = importlib.import_module("torch")
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    _torch = None
    TORCH_AVAILABLE = False

try:
    _sf = importlib.import_module("soundfile")
    SOUNDFILE_AVAILABLE = True
except ModuleNotFoundError:
    _sf = None
    SOUNDFILE_AVAILABLE = False

try:
    _librosa = importlib.import_module("librosa")
    LIBROSA_AVAILABLE = True
except ModuleNotFoundError:
    _librosa = None
    LIBROSA_AVAILABLE = False

ELEVENLABS_AVAILABLE = importlib.util.find_spec("elevenlabs") is not None

DIA_AVAILABLE = importlib.util.find_spec("dia") is not None

if TORCH_AVAILABLE and SOUNDFILE_AVAILABLE:
    try:
        _qwen_tts = importlib.import_module("qwen_tts")
        _Qwen3TTSModel = getattr(_qwen_tts, "Qwen3TTSModel", None)
        QWEN_TTS_AVAILABLE = _Qwen3TTSModel is not None
    except ModuleNotFoundError:
        _Qwen3TTSModel = None
        QWEN_TTS_AVAILABLE = False
else:
    _Qwen3TTSModel = None
    QWEN_TTS_AVAILABLE = False


class TTSService:
    """
    Comprehensive Text-to-Speech service supporting multiple providers
    """

    QWEN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    QWEN_DEFAULT_SPEAKER = "Ryan"
    QWEN_DEFAULT_LANGUAGE = "Auto"
    MAX_TTS_SPEEDUP_FACTOR = 1.4

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self._qwen_model = None
        self.gpu_manager = GPUMemoryManager(
            max_vram_usage=0.75
        )  # Increased for better TTS performance
        self._initialize_services()

    def _initialize_services(self):
        """Initialize available TTS services"""
        if QWEN_TTS_AVAILABLE:
            self.logger.info("Local Qwen3-TTS 1.7B CustomVoice model is available.")
        else:
            self.logger.warning(
                "Qwen3-TTS not available - install qwen-tts package dependencies"
            )

    def _initialize_qwen(self):
        """Lazy initialization of the local Qwen3-TTS model with GPU optimization."""
        if self._qwen_model is None and QWEN_TTS_AVAILABLE:
            try:
                torch_mod = _torch
                qwen_model_cls = _Qwen3TTSModel
                if torch_mod is None or qwen_model_cls is None:
                    self.logger.warning(
                        "Qwen3-TTS dependencies are not fully available"
                    )
                    return

                estimated_model_size_mb = 6000
                device = self.gpu_manager.get_optimal_device(estimated_model_size_mb)

                self.logger.info(f"Loading Qwen3-TTS model on {device}...")
                if device.startswith("cuda"):
                    with torch_mod.cuda.device(device):
                        torch_mod.cuda.empty_cache()

                    try:
                        self._qwen_model = qwen_model_cls.from_pretrained(
                            self.QWEN_MODEL_ID,
                            device_map=device,
                            dtype=torch_mod.bfloat16,
                            attn_implementation="flash_attention_2",
                        )
                    except Exception as flash_error:
                        self.logger.warning(
                            "FlashAttention initialization failed (%s), retrying without it",
                            flash_error,
                        )
                        self._qwen_model = qwen_model_cls.from_pretrained(
                            self.QWEN_MODEL_ID,
                            device_map=device,
                            dtype=torch_mod.bfloat16,
                        )

                    if hasattr(self._qwen_model, "model"):
                        self._qwen_model.model = (
                            self.gpu_manager.optimize_model_for_inference(
                                self._qwen_model.model
                            )
                        )

                else:
                    self.logger.info("Loading Qwen3-TTS model to CPU")
                    self._qwen_model = qwen_model_cls.from_pretrained(
                        self.QWEN_MODEL_ID,
                        device_map="cpu",
                        dtype=torch_mod.float32,
                    )

                self.logger.info("Qwen3-TTS 1.7B CustomVoice model loaded successfully")

            except Exception as e:
                self.logger.error(f"Failed to load Qwen3-TTS model: {e}")
                self._qwen_model = None
                if (
                    TORCH_AVAILABLE
                    and _torch is not None
                    and _torch.cuda.is_available()
                ):
                    _torch.cuda.empty_cache()

    def generate_speech(
        self, segment: NarrativeSegment, output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Generate speech for a narrative segment

        Args:
            segment: NarrativeSegment with text and parameters
            output_path: Optional output path for audio file

        Returns:
            Path to generated audio file or None if failed
        """
        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = Path(tmp.name)

        if QWEN_TTS_AVAILABLE:
            result = self._generate_with_qwen(segment, output_path)
            if result:
                return result

        try:
            pyttsx3_module = importlib.import_module("pyttsx3")
            return self._generate_with_pyttsx3(segment, output_path, pyttsx3_module)
        except ModuleNotFoundError:
            pass

        self.logger.error("No TTS service available for speech generation")
        return None

    def _generate_with_qwen(
        self, segment: NarrativeSegment, output_path: Path
    ) -> Optional[Path]:
        """Generate speech using the local Qwen3-TTS 1.7B CustomVoice model."""
        try:
            torch_mod = _torch
            sf_mod = _sf
            if torch_mod is None or sf_mod is None:
                self.logger.warning("Qwen3-TTS dependencies are not fully available")
                return None

            self._initialize_qwen()

            if self._qwen_model is None:
                self.logger.warning(
                    "Qwen3-TTS model not initialized, cannot generate speech."
                )
                return None

            instruct_text = self._build_qwen_instruction(segment)

            with torch_mod.inference_mode():
                wavs, sr = self._qwen_model.generate_custom_voice(
                    text=segment.text,
                    language=self.QWEN_DEFAULT_LANGUAGE,
                    speaker=self.QWEN_DEFAULT_SPEAKER,
                    instruct=instruct_text,
                    max_new_tokens=2048,
                )

            if not wavs:
                self.logger.warning("Qwen3-TTS returned no audio")
                return None

            audio_data = wavs[0]
            if hasattr(audio_data, "cpu"):
                audio_data = audio_data.cpu().numpy()

            if len(audio_data.shape) > 1:
                audio_data = audio_data.squeeze()

            sf_mod.write(str(output_path), audio_data, int(sr))

            if TORCH_AVAILABLE and torch_mod.cuda.is_available():
                torch_mod.cuda.empty_cache()
                gc.collect()

            self.logger.info(f"Generated Qwen3-TTS: '{segment.text[:30]}...'")
            return output_path

        except Exception as e:
            self.logger.warning(f"Local Qwen3-TTS failed, will try fallback TTS: {e}")
            if TORCH_AVAILABLE and _torch is not None and _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            return None

    def _build_qwen_instruction(self, segment: NarrativeSegment) -> str:
        """Build instruction text for Qwen3-TTS style control."""
        emotion = str(segment.emotion).lower()
        pacing = str(segment.pacing).lower()

        instructions = [
            "Narrate this clearly for a short-form video.",
            "Use a natural, human conversational voice.",
        ]

        if emotion == EmotionType.EXCITED.value:
            instructions.append("Sound energetic and excited.")
        elif emotion == EmotionType.DRAMATIC.value:
            instructions.append("Sound dramatic and suspenseful.")
        elif emotion == EmotionType.CALM.value:
            instructions.append("Sound calm and composed.")
        else:
            instructions.append("Sound neutral and clear.")

        if pacing == PacingType.FAST.value:
            instructions.append("Speak slightly faster than normal.")
        elif pacing == PacingType.SLOW.value:
            instructions.append("Speak slowly with clear pauses.")
        else:
            instructions.append("Speak at a natural pace.")

        return " ".join(instructions)

    def generate_multiple_segments(
        self, segments: List[NarrativeSegment], output_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate speech for multiple narrative segments

        Args:
            segments: List of NarrativeSegment objects
            output_dir: Directory to save audio files

        Returns:
            List of dictionaries with segment info and audio paths
        """
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="tts_"))
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for i, segment in enumerate(segments):
            try:
                output_path = output_dir / f"segment_{i:03d}.wav"
                audio_path = self.generate_speech(segment, output_path)

                result = {
                    "segment": segment,
                    "audio_path": audio_path,
                    "index": i,
                    "success": audio_path is not None,
                }

                if audio_path:
                    self._trim_segment_silence(audio_path, segment_index=i)

                    # FORCE fit to intended duration to prevent overlapping multiple TTS speaking at once
                    adjusted_path = self.adjust_audio_speed(
                        audio_path, segment.intended_duration_seconds
                    )
                    if adjusted_path and adjusted_path.exists():
                        # Clean up the original temporary file after creating the adjusted version
                        if adjusted_path != audio_path:
                            try:
                                Path(audio_path).unlink(missing_ok=True)
                            except Exception:
                                pass  # Best effort cleanup
                        audio_path = adjusted_path

                    # Get audio duration for timing validation
                    try:
                        clip = AudioFileClip(str(audio_path))
                        actual_duration = clip.duration
                        clip.close()

                        # FIX: Prevent overlapping TTS by forcing audio to fit intended duration
                        if actual_duration > segment.intended_duration_seconds:
                            speed_factor = (
                                actual_duration / segment.intended_duration_seconds
                            )
                            if speed_factor <= self.MAX_TTS_SPEEDUP_FACTOR:
                                adjusted_path = self.adjust_audio_speed(
                                    audio_path,
                                    target_duration=segment.intended_duration_seconds,
                                    output_path=audio_path.with_suffix(".fit.wav"),
                                )
                                if adjusted_path and adjusted_path.exists():
                                    audio_path = adjusted_path
                                    adj_clip = AudioFileClip(str(adjusted_path))
                                    try:
                                        actual_duration = adj_clip.duration
                                    finally:
                                        adj_clip.close()
                            else:
                                self.logger.warning(
                                    f"Segment {i} requires speedup of {speed_factor:.2f}x "
                                    f"(>{self.MAX_TTS_SPEEDUP_FACTOR}x), applying max speedup and truncating to fit"
                                )
                                max_target_duration = actual_duration / self.MAX_TTS_SPEEDUP_FACTOR
                                adjusted_path = self.adjust_audio_speed(
                                    audio_path,
                                    target_duration=max_target_duration,
                                    output_path=audio_path.with_suffix(".fit.wav"),
                                )
                                if adjusted_path and adjusted_path.exists():
                                    audio_path = adjusted_path

                                    adj_clip = AudioFileClip(str(adjusted_path))
                                    try:
                                        if adj_clip.duration > segment.intended_duration_seconds:
                                            # Truncate to prevent overlap
                                            trunc_clip = adj_clip.subclip(0, segment.intended_duration_seconds)
                                            trunc_path = adjusted_path.with_suffix(".trunc.wav")
                                            trunc_clip.write_audiofile(str(trunc_path), logger=None)
                                            audio_path = trunc_path
                                            trunc_clip.close()
                                    finally:
                                        adj_clip.close()

                                    actual_duration = segment.intended_duration_seconds

                        result["actual_duration"] = actual_duration
                        result["audio_path"] = audio_path
                    except Exception as e:
                        self.logger.warning(
                            f"Could not get/adjust duration for segment {i}: {e}"
                        )
                        result["actual_duration"] = segment.intended_duration_seconds

                results.append(result)

            except Exception as e:
                self.logger.error(f"Failed to generate TTS for segment {i}: {e}")
                results.append(
                    {
                        "segment": segment,
                        "audio_path": None,
                        "index": i,
                        "success": False,
                        "error": str(e),
                    }
                )

        successful = sum(1 for r in results if r["success"])
        self.logger.info(f"Generated TTS for {successful}/{len(segments)} segments")

        return results

    def _trim_segment_silence(self, audio_path: Path, segment_index: int = -1) -> None:
        """Trim leading/trailing silence conservatively for generated speech."""
        if not audio_path or not audio_path.exists():
            return

        try:
            if (
                LIBROSA_AVAILABLE
                and SOUNDFILE_AVAILABLE
                and _librosa is not None
                and _sf is not None
            ):
                if self._trim_silence_with_librosa(audio_path):
                    self.logger.debug(
                        "Trimmed silence for segment %s using librosa", segment_index
                    )
                    return
        except Exception as e:
            self.logger.debug(
                "Librosa silence trim unavailable for segment %s: %s",
                segment_index,
                e,
            )

        try:
            if self._trim_silence_with_moviepy(audio_path):
                self.logger.debug(
                    "Trimmed silence for segment %s using moviepy fallback",
                    segment_index,
                )
        except Exception as e:
            self.logger.warning(
                "Silence trim skipped for segment %s: %s", segment_index, e
            )

    def _trim_silence_with_librosa(self, audio_path: Path) -> bool:
        """Trim silence via librosa/soundfile for best waveform control."""
        if not (
            LIBROSA_AVAILABLE
            and SOUNDFILE_AVAILABLE
            and _librosa is not None
            and _sf is not None
        ):
            return False

        audio_data, sample_rate = _librosa.load(str(audio_path), sr=None, mono=False)
        if audio_data is None:
            return False

        _, trim_indices = _librosa.effects.trim(
            audio_data,
            top_db=28,
            frame_length=2048,
            hop_length=512,
        )
        start_idx, end_idx = int(trim_indices[0]), int(trim_indices[1])
        total_samples = audio_data.shape[-1]

        if end_idx <= start_idx:
            return False

        pad_samples = int(sample_rate * 0.04)
        start_idx = max(0, start_idx - pad_samples)
        end_idx = min(total_samples, end_idx + pad_samples)

        if start_idx <= 0 and end_idx >= total_samples:
            return False

        trimmed_audio = audio_data[..., start_idx:end_idx]
        if trimmed_audio.size == 0:
            return False

        if trimmed_audio.ndim > 1:
            trimmed_to_write = trimmed_audio.T
        else:
            trimmed_to_write = trimmed_audio

        _sf.write(str(audio_path), trimmed_to_write, int(sample_rate))
        return True

    def _trim_silence_with_moviepy(self, audio_path: Path) -> bool:
        """Fallback silence trim using moviepy + numpy when librosa is unavailable."""
        clip = AudioFileClip(str(audio_path))
        array_clip: Optional[Any] = None

        try:
            sample_rate = int(getattr(clip, "fps", 22050) or 22050)
            audio_array = clip.to_soundarray(fps=sample_rate)

            if audio_array is None or len(audio_array) == 0:
                return False

            if audio_array.ndim == 1:
                audio_array = audio_array.reshape(-1, 1)

            amplitude = np.max(np.abs(audio_array), axis=1)
            active = np.where(amplitude > 0.015)[0]
            if active.size == 0:
                return False

            start_idx = int(active[0])
            end_idx = int(active[-1]) + 1
            pad_samples = int(sample_rate * 0.04)

            start_idx = max(0, start_idx - pad_samples)
            end_idx = min(audio_array.shape[0], end_idx + pad_samples)

            if start_idx <= 0 and end_idx >= audio_array.shape[0]:
                return False

            trimmed = audio_array[start_idx:end_idx]
            if trimmed.size == 0:
                return False

            from moviepy.audio.AudioClip import AudioArrayClip

            array_clip = AudioArrayClip(trimmed.astype(np.float32), fps=sample_rate)
            write_audiofile = getattr(array_clip, "write_audiofile", None)
            if not callable(write_audiofile):
                return False

            write_audiofile(
                str(audio_path), fps=sample_rate, codec="pcm_s16le", logger=None
            )
            return True
        finally:
            if array_clip is not None:
                close_array = getattr(array_clip, "close", None)
                if callable(close_array):
                    close_array()
            clip.close()

    def adjust_audio_speed(
        self,
        audio_path: Path,
        target_duration: float,
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
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
            output_path = audio_path.with_suffix(".adjusted.wav")

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

            # Apply speed adjustment with MoviePy API compatibility
            with_speed_scaled = getattr(clip, "with_speed_scaled", None)
            if callable(with_speed_scaled):
                adjusted_clip = with_speed_scaled(speed_factor)
            else:
                fx_method = getattr(clip, "fx", None)
                if callable(fx_method):

                    def _speed_adjust_fallback(c):
                        set_duration = getattr(c, "set_duration", None)
                        if callable(set_duration):
                            return set_duration(c.duration / speed_factor)
                        with_duration = getattr(c, "with_duration", None)
                        if callable(with_duration):
                            return with_duration(c.duration / speed_factor)
                        return c

                    adjusted_clip = fx_method(_speed_adjust_fallback)
                else:
                    with_duration = getattr(clip, "with_duration", None)
                    if callable(with_duration):
                        adjusted_clip = with_duration(clip.duration / speed_factor)
                    else:
                        adjusted_clip = clip

            write_audiofile = getattr(adjusted_clip, "write_audiofile", None)
            if callable(write_audiofile):
                write_audiofile(str(output_path), verbose=False, logger=None)
            else:
                self.logger.warning(
                    "Adjusted clip missing write_audiofile, returning original audio"
                )
                clip.close()
                return audio_path

            clip.close()
            close_adjusted = getattr(adjusted_clip, "close", None)
            if callable(close_adjusted):
                close_adjusted()

            self.logger.debug(
                f"Adjusted audio {audio_path.name} speed by factor {speed_factor:.2f}"
            )
            return output_path

        except Exception as e:
            self.logger.warning(f"Failed to adjust audio speed: {e}")
            return audio_path

    def is_available(self) -> bool:
        """Check if any TTS service is available"""
        return QWEN_TTS_AVAILABLE

    def get_available_services(self) -> List[str]:
        """Get list of available TTS services"""
        services = []
        if QWEN_TTS_AVAILABLE:
            services.append("qwen3_tts")
        return services

    def _generate_with_pyttsx3(
        self, segment: NarrativeSegment, output_path: Path, pyttsx3_module: Any
    ) -> Optional[Path]:
        """Generate speech using system TTS (pyttsx3) as fallback."""
        try:
            engine = pyttsx3_module.init()

            # Configure voice settings based on emotion and pacing
            rate = 200  # Default rate
            if segment.pacing == PacingType.SLOW:
                rate = 150
            elif segment.pacing == PacingType.FAST:
                rate = 250

            engine.setProperty("rate", rate)
            engine.setProperty("volume", 0.9)

            # Save to file
            engine.save_to_file(segment.text, str(output_path))
            engine.runAndWait()

            self.logger.info(f"Generated pyttsx3 TTS: '{segment.text[:30]}...'")
            return output_path

        except Exception as e:
            self.logger.warning(f"pyttsx3 TTS failed: {e}")
            return None
