"""
Configuration management for YouTube video generation system.
Centralizes all configuration from multiple sources: config.yaml, .env, and defaults.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, List, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file, but don't override existing system variables
load_dotenv(override=False)


@dataclass
class VideoConfig:
    """Video processing configuration"""

    target_duration: int = 59
    target_resolution: tuple = (1080, 1920)
    target_aspect_ratio: float = 9 / 16
    target_fps: int = 30
    max_retries: int = 3
    default_crop: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.75, 0.75])

    # Audio settings
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    loudness_target_lufs: float = -14.0
    default_audio_fps: int = 44100

    # Video encoding
    video_codec_cpu: str = "libx264"
    video_codec_gpu: str = "h264_nvenc"
    ffmpeg_cpu_preset: str = "medium"
    ffmpeg_gpu_preset: str = "p5"
    ffmpeg_crf_cpu: str = "23"
    ffmpeg_cq_gpu: str = "23"
    video_bitrate_high: str = "10M"

    # GPU Memory Optimization
    gpu_memory_fraction: float = 0.6  # Use 60% of VRAM for video processing
    enable_gpu_acceleration: bool = True
    nvenc_memory_pool_size: str = "512M"  # Limit NVENC memory pool

    # Quality profile for easy management
    video_quality_profile: str = (
        "standard"  # Options: 'standard', 'high', 'maximum', 'speed'
    )

    # Speed optimization settings
    enable_speed_optimization: bool = True
    speed_optimization_level: str = (
        "aggressive"  # Options: 'conservative', 'balanced', 'aggressive'
    )

    # Processing
    chunk_size: int = 30  # seconds
    max_memory_usage: float = 0.8  # 80% of available memory
    max_vram_usage: float = 0.9  # 90% of available VRAM


@dataclass
class TextOverlayConfig:
    """Text overlay and subtitle configuration"""

    # Graphical text (from AI analysis)
    graphical_font: str = "Montserrat-Bold.ttf"
    graphical_font_size_ratio: float = 1 / 18
    graphical_text_color: str = "white"
    graphical_stroke_color: str = "black"
    graphical_stroke_width: int = 2
    graphical_bg_color: str = "rgba(0,0,0,0.5)"

    # Subtitles
    subtitle_font: str = "Montserrat-Regular.ttf"
    subtitle_font_size_ratio: float = 1 / 25
    subtitle_text_color: str = "white"
    subtitle_stroke_color: str = "black"
    subtitle_stroke_width: int = 1
    subtitle_position: tuple = ("center", 0.92)
    subtitle_bg_color: str = "rgba(0,0,0,0.4)"

    # Font profiles by length
    font_size_ratio_profiles: Dict[str, float] = field(
        default_factory=lambda: {"short": 0.06, "medium": 0.045, "long": 0.035}
    )

    # Animation settings
    fade_in: float = 0.2
    fade_out: float = 0.2
    animation: str = "fade"


@dataclass
class EffectsConfig:
    """Video effects configuration"""

    shake_intensity: float = 0.02
    max_zoom: float = 1.1
    stabilization_threshold: float = 5.0
    color_grade_intensity: float = 0.7
    subtle_zoom_enabled: bool = True
    color_grade_enabled: bool = True

    # Seamless looping
    enable_seamless_looping: bool = True
    loop_crossfade_duration: float = 0.3
    loop_compatibility_threshold: float = 0.3
    loop_sample_duration: float = 0.5
    loop_target_duration: Optional[float] = None
    enable_audio_crossfade: bool = True
    loop_extend_mode: str = "middle_repeat"
    loop_trim_from_center: bool = True


@dataclass
class AudioConfig:
    """Audio processing configuration"""

    original_audio_mix_volume: float = 0.15
    background_music_enabled: bool = True
    background_music_volume: float = 0.06
    background_music_narrative_volume_factor: float = 0.2

    # Music categories and presets
    music_categories: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "upbeat": ["energetic", "positive", "happy", "uplifting"],
            "emotional": ["sad", "heartwarming", "touching", "sentimental"],
            "suspenseful": ["tense", "dramatic", "action", "exciting"],
            "relaxing": ["calm", "peaceful", "ambient", "soothing"],
            "funny": ["quirky", "comedic", "playful", "lighthearted"],
            "informative": ["neutral", "documentary", "educational", "background"],
        }
    )


@dataclass
class APIConfig:
    """API configuration for external services"""

    # Reddit
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "python:VideoBot:v1.8 (by /u/YOUR_USERNAME)"

    # YouTube
    youtube_api_service_name: str = "youtube"
    youtube_api_version: str = "v3"
    youtube_scopes: List[str] = field(
        default_factory=lambda: [
            "https://www.googleapis.com/auth/youtube.upload",
            "https://www.googleapis.com/auth/youtube.force-ssl",
            "https://www.googleapis.com/auth/youtubepartner",
        ]
    )
    youtube_upload_category_id: str = "24"
    youtube_upload_privacy_status: str = "public"
    youtube_self_certification: bool = True
    # OAuth / token file paths (may be set from config.yaml `api` section)
    youtube_client_secrets_file: str = ""
    youtube_token_file: str = ""

    # NVIDIA NIM
    nvidia_nim_api_key: str = ""
    nvidia_nim_model: str = "qwen/qwen2.5-7b-instruct"
    nvidia_nim_alt_model: str = "qwen/qwen3-235b-a22b"
    nvidia_nim_rate_limit_rpm: int = 60
    nvidia_nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    ai_provider: str = "nvidia_nim"

    # ElevenLabs TTS
    elevenlabs_timeout: int = 20
    default_voice_id: str = "21m00Tcm4TlvDq8ikWAM"

    # Rate limiting
    api_delay_seconds: int = 2


@dataclass
class ContentConfig:
    """Content filtering and curation configuration"""

    max_reddit_posts_to_fetch: int = 10

    curated_subreddits: List[str] = field(
        default_factory=lambda: [
            "oddlysatisfying",
            "nextfuckinglevel",
            "BeAmazed",
            "woahdude",
            "MadeMeSmile",
            "Eyebleach",
            "interestingasfuck",
            "Damnthatsinteresting",
            "AnimalsBeingBros",
            "HumansBeingBros",
            "wholesomememes",
            "ContagiousLaughter",
            "foodporn",
            "CookingVideos",
            "ArtisanVideos",
            "educationalgifs",
            "DIY",
            "gardening",
            "science",
            "space",
            "NatureIsCool",
            "aww",
            "AnimalsBeingDerps",
            "rarepuppers",
            "LifeProTips",
            "GetMotivated",
            "toptalent",
            "BetterEveryLoop",
            "childrenfallingover",
            "instantregret",
            "wholesomegifs",
            "Unexpected",
            "nevertellmetheodds",
            "whatcouldgoright",
            "holdmymilk",
            "maybemaybemaybe",
            "mildlyinteresting",
        ]
    )

    forbidden_words: List[str] = field(
        default_factory=lambda: [
            "fuck",
            "fucking",
            "fucked",
            "fuckin",
            "shit",
            "shitty",
            "shitting",
            "wtf",
            "stfu",
            "omfg",
            "porn",
            "pornographic",
            "nsfw",
            "xxx",
            "sex",
            "sexual",
            "nude",
            "naked",
            "racist",
            "racism",
            "nazi",
            "sexist",
            "dangerous activities glorification",
            "suicide promotion",
            "self-harm",
        ]
    )

    hard_disallowed: List[str] = field(
        default_factory=lambda: [
            "porn",
            "pornographic",
            "nsfw",
            "xxx",
            "sexual",
            "nude",
            "naked",
            "racist",
            "racism",
            "nazi",
            "sexist",
            "suicide promotion",
            "self-harm",
        ]
    )

    demonetization_risk: List[str] = field(
        default_factory=lambda: [
            "violence",
            "violent",
            "gore",
            "blood",
            "death",
            "killing",
            "murder",
            "weapon",
            "gun",
            "knife",
            "drug",
            "drugs",
            "cocaine",
            "heroin",
            "weed",
            "smoking",
            "alcohol",
            "drunk",
            "gambling",
            "casino",
            "betting",
            "suicide",
            "mental health crisis",
            "adult content",
            "mature content",
            "18+",
            "not safe for work",
        ]
    )

    caution: List[str] = field(
        default_factory=lambda: [
            "political",
            "politics",
            "election",
            "religion",
            "religious",
            "conspiracy",
            "fake news",
            "misinformation",
            "hoax",
            "accident",
            "crash",
            "injury",
            "medical emergency",
            "depression",
        ]
    )

    unsuitable_content_types: List[str] = field(
        default_factory=lambda: [
            "violence",
            "violent",
            "gore",
            "blood",
            "death",
            "killing",
            "murder",
            "weapon",
            "gun",
            "knife",
            "drug",
            "drugs",
            "cocaine",
            "heroin",
            "marijuana",
            "weed",
            "smoking",
            "alcohol",
            "drunk",
            "gambling",
            "casino",
            "bet",
            "betting",
            "political",
            "politics",
            "election",
            "trump",
            "biden",
            "religion",
            "religious",
            "god",
            "jesus",
            "muslim",
            "christian",
            "church",
            "mosque",
            "suicide",
            "depression",
            "self-harm",
            "cutting",
            "mental health crisis",
            "accident",
            "crash",
            "injury",
            "hospital",
            "medical emergency",
            "conspiracy",
            "fake news",
            "misinformation",
            "hoax",
            "adult content",
            "mature content",
            "18+",
            "nsfw",
            "not safe for work",
        ]
    )

    monetization_tags: List[str] = field(
        default_factory=lambda: [
            "family friendly",
            "educational",
            "informative",
            "wholesome",
            "positive",
            "interesting",
            "amazing",
            "satisfying",
            "relaxing",
            "shorts",
            "shortvideo",
        ]
    )


@dataclass
class PathConfig:
    """File and directory path configuration"""

    base_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent.resolve()
    )
    temp_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent.resolve()
        / "data"
        / "temp"
    )
    processed_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent.resolve()
        / "processed"
    )

    # Asset directories
    music_folder: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent.resolve() / "music"
    )
    sound_effects_folder: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent.resolve()
        / "sound_effects"
    )
    fonts_folder: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent.resolve() / "fonts"
    )
    thumbnails_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent.resolve()
        / "thumbnails"
    )
    cache_folder: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent.resolve()
        / "data"
        / "cache"
    )

    # Files
    db_file: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent.resolve()
        / "data"
        / "databases"
        / "uploaded_videos.db"
    )
    watermark_path: Optional[Path] = None

    # Credentials and tokens
    google_client_secrets_file: Optional[Path] = None
    youtube_token_file: Optional[Path] = None


class ConfigManager:
    """Load and validate settings from config.yaml, .env, and defaults.

    YAML: only matching dataclass field names are applied (see `_load_yaml_config`).
    Sections such as ``ai_features`` or ``channel_management`` are ignored at
    runtime but may remain in ``config.yaml`` for documentation or because
    ``EnhancementOptimizer`` (``src/processing/enhancement_optimizer.py``)
    round-trips the full file when updating tunable parameters.
    """

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        self.config_file = Path(config_file) if config_file else None
        self.logger = logging.getLogger(__name__)

        self._reset_to_defaults()

        # Load configurations in order of precedence
        self._load_yaml_config()
        self._load_env_config()
        self._validate_config()

    def _reset_to_defaults(self):
        """Reset all configuration sections to default values."""
        # Initialize default configurations
        self.video = VideoConfig()
        self.text_overlay = TextOverlayConfig()
        self.effects = EffectsConfig()
        self.audio = AudioConfig()
        self.api = APIConfig()
        self.content = ContentConfig()
        self.paths = PathConfig()

    def reload(self) -> "ConfigManager":
        """Reload configuration from current config file and environment."""
        self._reset_to_defaults()
        self._load_yaml_config()
        self._load_env_config()
        self._validate_config()
        self.logger.info("Configuration reloaded")
        return self

    def _load_yaml_config(self):
        """Load configuration from YAML file"""
        if not self.config_file:
            # Try to find config.yaml in the project root
            potential_configs = [
                Path("config.yaml"),
                Path("../config.yaml"),
                Path("../../config.yaml"),
                self.paths.base_dir / "config.yaml",
            ]

            for config_path in potential_configs:
                if config_path.exists():
                    self.config_file = config_path
                    break

        if not self.config_file or not self.config_file.exists():
            self.logger.warning(
                "No config.yaml found, using defaults and environment variables"
            )
            return

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)

            if not yaml_config:
                return

            self._apply_yaml_paths_and_database(yaml_config)

            # Update configurations from YAML
            self._update_from_dict(self.video, yaml_config.get("video", {}))
            # Also check for video_processing section
            self._update_from_dict(self.video, yaml_config.get("video_processing", {}))
            self._update_from_dict(
                self.text_overlay, yaml_config.get("text_overlay", {})
            )
            self._update_from_dict(self.effects, yaml_config.get("effects", {}))
            self._update_from_dict(self.audio, yaml_config.get("audio", {}))
            self._update_from_dict(self.api, yaml_config.get("apis", {}))
            self._update_from_dict(self.api, yaml_config.get("api", {}))
            self._resolve_credential_paths_from_api()
            self._update_from_dict(self.content, yaml_config.get("content", {}))

            # Handle nested audio keys (e.g., audio.background_music.volume)
            audio_config = yaml_config.get("audio", {})
            if isinstance(audio_config, dict):
                background_music = audio_config.get("background_music", {})
                if isinstance(background_music, dict) and "volume" in background_music:
                    self.audio.background_music_volume = background_music["volume"]

            # Handle subtitles config specifically
            subtitles_config = yaml_config.get("subtitles", {})
            if subtitles_config:
                if "font_size_ratio_profiles" in subtitles_config:
                    self.text_overlay.font_size_ratio_profiles.update(
                        subtitles_config["font_size_ratio_profiles"]
                    )
                self._update_from_dict(self.text_overlay, subtitles_config)

            # Handle looping config
            looping_config = yaml_config.get("looping", {})
            if looping_config:
                self._update_from_dict(self.effects, looping_config)

            # Handle subreddits list
            if "subreddits" in yaml_config:
                self.content.curated_subreddits = yaml_config["subreddits"]

            # Handle fallbacks
            fallbacks = yaml_config.get("fallbacks", {})
            if fallbacks:
                self._update_from_dict(self.effects, fallbacks)

            self.logger.info(f"Loaded configuration from {self.config_file}")

        except Exception as e:
            self.logger.error(f"Error loading YAML config: {e}")

    def _config_yaml_parent(self) -> Path:
        if self.config_file and self.config_file.exists():
            return self.config_file.resolve().parent
        return Path.cwd()

    def _resolve_path_value(self, raw: Union[str, Path], base: Path) -> Path:
        p = Path(raw)
        return (base / p).resolve() if not p.is_absolute() else p.resolve()

    def _apply_yaml_paths_and_database(self, yaml_config: Dict) -> None:
        """Apply `paths` and `database` sections so PathConfig matches config.yaml."""
        paths_cfg = yaml_config.get("paths") or {}
        cfg_parent = self._config_yaml_parent()

        if paths_cfg.get("base_dir") is not None and str(paths_cfg.get("base_dir", "")).strip():
            self.paths.base_dir = self._resolve_path_value(paths_cfg["base_dir"], cfg_parent)

        base = self.paths.base_dir

        mapping = (
            ("temp_dir", "temp_dir"),
            ("processed_dir", "processed_dir"),
            ("fonts_dir", "fonts_folder"),
            ("music_dir", "music_folder"),
            ("sound_effects_dir", "sound_effects_folder"),
            ("thumbnails_dir", "thumbnails_dir"),
            ("ai_models_cache_dir", "cache_folder"),
        )
        for yaml_key, attr in mapping:
            val = paths_cfg.get(yaml_key)
            if val is not None and str(val).strip():
                setattr(self.paths, attr, self._resolve_path_value(val, base))

        db_cfg = yaml_config.get("database") or {}
        if db_cfg.get("sqlite_db_path"):
            self.paths.db_file = self._resolve_path_value(
                db_cfg["sqlite_db_path"], base
            )

    def _resolve_credential_paths_from_api(self) -> None:
        """Map api.youtube_* file entries from YAML onto PathConfig."""
        base = self.paths.base_dir
        secrets = (self.api.youtube_client_secrets_file or "").strip()
        if secrets:
            self.paths.google_client_secrets_file = self._resolve_path_value(
                secrets, base
            )
        token = (self.api.youtube_token_file or "").strip()
        if token:
            self.paths.youtube_token_file = self._resolve_path_value(token, base)

    def _load_env_config(self):
        """Load configuration from environment variables"""
        # API credentials - only set if not empty to avoid overriding with empty strings
        reddit_client_id = os.getenv("REDDIT_CLIENT_ID", "").strip()
        if reddit_client_id:
            self.api.reddit_client_id = reddit_client_id

        reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
        if reddit_client_secret:
            self.api.reddit_client_secret = reddit_client_secret

        reddit_user_agent = os.getenv("REDDIT_USER_AGENT", "").strip()
        if reddit_user_agent:
            self.api.reddit_user_agent = reddit_user_agent

        nvidia_nim_api_key = os.getenv("NVIDIA_NIM_API_KEY", "").strip()
        if nvidia_nim_api_key:
            self.api.nvidia_nim_api_key = nvidia_nim_api_key

        nvidia_nim_model = os.getenv("NVIDIA_NIM_MODEL", "").strip()
        if nvidia_nim_model:
            self.api.nvidia_nim_model = nvidia_nim_model

        nvidia_nim_alt_model = os.getenv("NVIDIA_NIM_ALT_MODEL", "").strip()
        if nvidia_nim_alt_model:
            self.api.nvidia_nim_alt_model = nvidia_nim_alt_model

        nvidia_nim_base_url = os.getenv("NVIDIA_NIM_BASE_URL", "").strip()
        if nvidia_nim_base_url:
            self.api.nvidia_nim_base_url = nvidia_nim_base_url

        nvidia_nim_rate_limit_rpm = os.getenv("NVIDIA_NIM_RATE_LIMIT_RPM", "").strip()
        if nvidia_nim_rate_limit_rpm:
            try:
                self.api.nvidia_nim_rate_limit_rpm = int(nvidia_nim_rate_limit_rpm)
            except ValueError:
                self.logger.warning(
                    "Invalid NVIDIA_NIM_RATE_LIMIT_RPM value '%s'; using default %s",
                    nvidia_nim_rate_limit_rpm,
                    self.api.nvidia_nim_rate_limit_rpm,
                )

        ai_provider = os.getenv("AI_PROVIDER", "").strip().lower()
        if ai_provider:
            self.api.ai_provider = ai_provider

        # File paths
        if os.getenv("GOOGLE_CLIENT_SECRETS_FILE"):
            self.paths.google_client_secrets_file = Path(
                os.getenv("GOOGLE_CLIENT_SECRETS_FILE")
            )

        if os.getenv("YOUTUBE_TOKEN_FILE"):
            self.paths.youtube_token_file = Path(os.getenv("YOUTUBE_TOKEN_FILE"))

        if os.getenv("DB_FILE_PATH"):
            self.paths.db_file = Path(os.getenv("DB_FILE_PATH"))

        if os.getenv("WATERMARK_FILE_PATH"):
            self.paths.watermark_path = Path(os.getenv("WATERMARK_FILE_PATH"))

        if os.getenv("MUSIC_FILES_DIR"):
            self.paths.music_folder = Path(os.getenv("MUSIC_FILES_DIR"))

        if os.getenv("SOUND_EFFECTS_DIR"):
            self.paths.sound_effects_folder = Path(os.getenv("SOUND_EFFECTS_DIR"))

        # Looping configuration from environment
        self.effects.enable_seamless_looping = (
            os.getenv("ENABLE_SEAMLESS_LOOPING", "true").lower() == "true"
        )
        loop_crossfade_duration = os.getenv(
            "LOOP_CROSSFADE_DURATION", str(self.effects.loop_crossfade_duration)
        )
        try:
            self.effects.loop_crossfade_duration = float(loop_crossfade_duration)
        except ValueError:
            self.logger.warning(
                "Invalid LOOP_CROSSFADE_DURATION '%s'; using %s",
                loop_crossfade_duration,
                self.effects.loop_crossfade_duration,
            )

        loop_compatibility_threshold = os.getenv(
            "LOOP_COMPATIBILITY_THRESHOLD",
            str(self.effects.loop_compatibility_threshold),
        )
        try:
            self.effects.loop_compatibility_threshold = float(
                loop_compatibility_threshold
            )
        except ValueError:
            self.logger.warning(
                "Invalid LOOP_COMPATIBILITY_THRESHOLD '%s'; using %s",
                loop_compatibility_threshold,
                self.effects.loop_compatibility_threshold,
            )

        loop_sample_duration = os.getenv(
            "LOOP_SAMPLE_DURATION", str(self.effects.loop_sample_duration)
        )
        try:
            self.effects.loop_sample_duration = float(loop_sample_duration)
        except ValueError:
            self.logger.warning(
                "Invalid LOOP_SAMPLE_DURATION '%s'; using %s",
                loop_sample_duration,
                self.effects.loop_sample_duration,
            )

        if os.getenv("LOOP_TARGET_DURATION"):
            loop_target_duration = os.getenv("LOOP_TARGET_DURATION", "")
            try:
                self.effects.loop_target_duration = float(loop_target_duration)
            except ValueError:
                self.logger.warning(
                    "Invalid LOOP_TARGET_DURATION '%s'; keeping %s",
                    loop_target_duration,
                    self.effects.loop_target_duration,
                )

        self.effects.enable_audio_crossfade = (
            os.getenv("ENABLE_AUDIO_CROSSFADE", "true").lower() == "true"
        )
        self.effects.loop_extend_mode = os.getenv(
            "LOOP_EXTEND_MODE", self.effects.loop_extend_mode
        )
        self.effects.loop_trim_from_center = (
            os.getenv("LOOP_TRIM_FROM_CENTER", "true").lower() == "true"
        )

        self.logger.info("Loaded environment variable configuration")

    def _update_from_dict(self, config_obj, config_dict):
        """Update dataclass from dictionary, preserving types"""
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                try:
                    setattr(config_obj, key, value)
                except (TypeError, ValueError) as e:
                    self.logger.warning(f"Failed to set {key}={value}: {e}")

    def _validate_config(self):
        """Validate critical configuration settings"""
        errors = []
        warnings = []

        # Check essential API credentials
        if not self.api.reddit_client_id:
            warnings.append(
                "REDDIT_CLIENT_ID not set - Reddit functionality will be disabled"
            )
        if not self.api.reddit_client_secret:
            warnings.append(
                "REDDIT_CLIENT_SECRET not set - Reddit functionality will be disabled"
            )
        if not self.api.nvidia_nim_api_key:
            warnings.append("NVIDIA_NIM_API_KEY not set - AI analysis will be disabled")

        if self.api.ai_provider != "nvidia_nim":
            warnings.append(
                f"Unsupported AI_PROVIDER '{self.api.ai_provider}' - falling back to nvidia_nim"
            )
            self.api.ai_provider = "nvidia_nim"

        # Check file paths
        if self.paths.youtube_token_file and not self.paths.youtube_token_file.exists():
            warnings.append(
                f"YouTube token file not found: {self.paths.youtube_token_file}"
            )

        if (
            self.paths.google_client_secrets_file
            and not self.paths.google_client_secrets_file.exists()
        ):
            warnings.append(
                f"Google client secrets file not found: {self.paths.google_client_secrets_file}"
            )

        # Ensure directories exist
        self.paths.temp_dir.mkdir(parents=True, exist_ok=True)
        self.paths.music_folder.mkdir(parents=True, exist_ok=True)
        self.paths.sound_effects_folder.mkdir(parents=True, exist_ok=True)
        self.paths.fonts_folder.mkdir(parents=True, exist_ok=True)
        self.paths.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        self.paths.cache_folder.mkdir(parents=True, exist_ok=True)

        # Log validation results
        if errors:
            error_msg = "Configuration validation errors: " + "; ".join(errors)
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if warnings:
            for warning in warnings:
                self.logger.warning(f"Configuration warning: {warning}")

        self.logger.info("Configuration validation completed")

    def get_font_path(self, font_name: str) -> str:
        """Get the full path to a font file"""
        font_path = self.paths.fonts_folder / font_name
        if font_path.exists():
            return str(font_path)

        # Fallback to system font
        self.logger.warning(f"Font {font_name} not found, using Arial")
        return "Arial"

    def get_music_path(self, filename: str) -> Optional[Path]:
        """Get the full path to a music file"""
        music_path = self.paths.music_folder / filename
        return music_path if music_path.exists() else None

    def get_sound_effect_path(self, filename: str) -> Optional[Path]:
        """Get the full path to a sound effect file"""
        effect_path = self.paths.sound_effects_folder / filename
        return effect_path if effect_path.exists() else None

    def log_config_summary(self):
        """Log a summary of the current configuration"""
        self.logger.info("=== Configuration Summary ===")
        self.logger.info(
            f"Video: {self.video.target_resolution[0]}x{self.video.target_resolution[1]} @ {self.video.target_fps}fps"
        )
        self.logger.info(
            f"Reddit: {'[OK]' if self.api.reddit_client_id else '[FAIL]'} configured"
        )
        self.logger.info(
            f"YouTube: {'[OK]' if self.paths.youtube_token_file and self.paths.youtube_token_file.exists() else '[FAIL]'} configured"
        )
        self.logger.info(
            f"NVIDIA NIM: {'[OK]' if self.api.nvidia_nim_api_key else '[FAIL]'} configured"
        )
        self.logger.info(f"AI provider: {self.api.ai_provider}")
        self.logger.info(
            f"Seamless looping: {'[ENABLED]' if self.effects.enable_seamless_looping else '[DISABLED]'}"
        )
        self.logger.info(
            f"Background music: {'[ENABLED]' if self.audio.background_music_enabled else '[DISABLED]'}"
        )
        self.logger.info(f"Base directory: {self.paths.base_dir}")
        self.logger.info(f"Temp directory: {self.paths.temp_dir}")
        self.logger.info("================================")


# Global configuration instance
config: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get the global configuration instance"""
    global config
    if config is None:
        config = ConfigManager()
    return config


def init_config(config_file: Optional[Union[str, Path]] = None) -> ConfigManager:
    """Initialize the global configuration"""
    global config
    config = ConfigManager(config_file)
    return config


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration with Unicode support"""
    import sys
    import codecs

    # Create handlers with proper encoding
    console_handler = logging.StreamHandler()

    # For Windows, wrap stdout to handle Unicode properly
    if sys.platform.startswith("win"):
        # Ensure console can handle UTF-8
        if hasattr(console_handler.stream, "reconfigure"):
            try:
                console_handler.stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                # If reconfigure fails, use a wrapper
                console_handler.stream = codecs.getwriter("utf-8")(
                    console_handler.stream.buffer, errors="replace"
                )
        else:
            # Fallback for older Python versions
            console_handler.stream = codecs.getwriter("utf-8")(
                console_handler.stream.buffer
                if hasattr(console_handler.stream, "buffer")
                else console_handler.stream,
                errors="replace",
            )

    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler("youtube_generator.log", encoding="utf-8")

    # Set format for both handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()), handlers=[console_handler, file_handler]
    )
