"""
Text-to-Speech service with support for local Qwen3-TTS 1.7B CustomVoice.
Handles dynamic TTS parameters, emotion, and pacing for engaging narration.
Optimized for GPU memory usage with constrained VRAM environments.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import gc
import importlib

_torch: Any = None
_sf: Any = None
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
from moviepy import AudioFileClip

from src.config.settings import get_config
from src.models import NarrativeSegment, EmotionType, PacingType
from src.utils.gpu_memory_manager import GPUMemoryManager


class TTSService:
    """
    Comprehensive Text-to-Speech service supporting multiple providers
    """

    QWEN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    QWEN_DEFAULT_SPEAKER = "Ryan"
    QWEN_DEFAULT_LANGUAGE = "Auto"

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
            output_path = Path(tempfile.mktemp(suffix=".wav"))

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
                    # Get audio duration for timing validation
                    try:
                        clip = AudioFileClip(str(audio_path))
                        result["actual_duration"] = clip.duration
                        clip.close()
                    except Exception as e:
                        self.logger.warning(
                            f"Could not get duration for segment {i}: {e}"
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
                    adjusted_clip = fx_method(
                        lambda c: c.set_duration(c.duration / speed_factor)
                    )
                else:
                    adjusted_clip = clip.with_duration(clip.duration / speed_factor)

            adjusted_clip.write_audiofile(str(output_path), verbose=False, logger=None)

            clip.close()
            adjusted_clip.close()

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
