import logging
import sys
import types
import importlib
from pathlib import Path

import yaml

from src.config.settings import ConfigManager, get_config, init_config
from src.database.db_manager import DatabaseManager
from src.processing.enhancement_optimizer import EnhancementOptimizer


def test_config_loads_api_and_apis_sections(tmp_path: Path):
    config_api = tmp_path / "config_api.yaml"
    config_api.write_text(
        """
api:
  youtube_api_version: v99
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config_apis = tmp_path / "config_apis.yaml"
    config_apis.write_text(
        """
apis:
  youtube_api_version: v42
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg_api = ConfigManager(config_file=config_api)
    cfg_apis = ConfigManager(config_file=config_apis)

    assert cfg_api.api.youtube_api_version == "v99"
    assert cfg_apis.api.youtube_api_version == "v42"


def test_enhancement_optimizer_updates_yaml_and_reloads(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
audio:
  background_music:
    volume: 0.2
  sound_effects:
    volume: 0.4
""".strip()
        + "\n",
        encoding="utf-8",
    )

    init_config(config_path)
    optimizer = EnhancementOptimizer()

    optimizer._update_config_parameters(
        {
            "background_music_volume": 0.55,
            "sound_effects_volume": 0.35,
        }
    )

    updated_yaml = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert updated_yaml["audio"]["background_music"]["volume"] == 0.55
    assert updated_yaml["audio"]["sound_effects"]["volume"] == 0.35

    refreshed_config = get_config()
    assert refreshed_config.audio.background_music_volume == 0.55


def test_config_reload_updates_global_config_view(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
audio:
  background_music:
    volume: 0.1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    init_config(config_path)
    cfg = get_config()
    assert cfg.audio.background_music_volume == 0.1

    config_path.write_text(
        """
audio:
  background_music:
    volume: 0.65
""".strip()
        + "\n",
        encoding="utf-8",
    )

    reloaded = cfg.reload()
    assert reloaded is cfg
    assert get_config().audio.background_music_volume == 0.65


def test_channel_manager_get_video_path_uses_db_mapping(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "artifacts.db"
    db_manager = DatabaseManager(db_path=db_path)

    video_path = tmp_path / "mapped_video.mp4"
    video_path.write_bytes(b"data")

    db_manager.record_local_artifacts(
        reddit_url="https://reddit.com/r/test/comments/abc123/post",
        local_project_id="project-abc123",
        local_video_path=str(video_path),
        local_thumbnail_paths=[str(tmp_path / "thumb.png")],
        youtube_video_id="yt_video_123",
    )

    fake_youtube_module = types.ModuleType("src.integrations.youtube_client")
    setattr(fake_youtube_module, "YouTubeClient", type("YouTubeClient", (), {}))

    fake_ai_module = types.ModuleType("src.integrations.ai_client")
    setattr(fake_ai_module, "AIClient", type("AIClient", (), {}))

    fake_thumbnail_module = types.ModuleType(
        "src.processing.enhanced_thumbnail_generator"
    )
    setattr(
        fake_thumbnail_module,
        "EnhancedThumbnailGenerator",
        type("EnhancedThumbnailGenerator", (), {}),
    )

    fake_engagement_module = types.ModuleType("src.monitoring.engagement_metrics")
    setattr(
        fake_engagement_module, "EngagementMonitor", type("EngagementMonitor", (), {})
    )

    monkeypatch.setitem(
        sys.modules, "src.integrations.youtube_client", fake_youtube_module
    )
    monkeypatch.setitem(sys.modules, "src.integrations.ai_client", fake_ai_module)
    monkeypatch.setitem(
        sys.modules,
        "src.processing.enhanced_thumbnail_generator",
        fake_thumbnail_module,
    )
    monkeypatch.setitem(
        sys.modules, "src.monitoring.engagement_metrics", fake_engagement_module
    )

    sys.modules.pop("src.management.channel_manager", None)
    channel_manager_module = importlib.import_module("src.management.channel_manager")
    ChannelManager = channel_manager_module.ChannelManager

    manager = ChannelManager.__new__(ChannelManager)
    manager.logger = logging.getLogger(__name__)
    manager._get_db_manager = lambda: db_manager

    resolved_path = manager._get_video_path("yt_video_123")
    assert resolved_path == video_path


def test_channel_manager_get_video_path_rejects_missing_local_file(
    tmp_path: Path, monkeypatch
):
    db_path = tmp_path / "artifacts.db"
    db_manager = DatabaseManager(db_path=db_path)

    missing_video_path = tmp_path / "missing_video.mp4"
    db_manager.record_local_artifacts(
        reddit_url="https://reddit.com/r/test/comments/abc124/post",
        local_project_id="project-abc124",
        local_video_path=str(missing_video_path),
        local_thumbnail_paths=[],
        youtube_video_id="yt_video_124",
    )

    fake_youtube_module = types.ModuleType("src.integrations.youtube_client")
    setattr(fake_youtube_module, "YouTubeClient", type("YouTubeClient", (), {}))

    fake_ai_module = types.ModuleType("src.integrations.ai_client")
    setattr(fake_ai_module, "AIClient", type("AIClient", (), {}))

    fake_thumbnail_module = types.ModuleType(
        "src.processing.enhanced_thumbnail_generator"
    )
    setattr(
        fake_thumbnail_module,
        "EnhancedThumbnailGenerator",
        type("EnhancedThumbnailGenerator", (), {}),
    )

    fake_engagement_module = types.ModuleType("src.monitoring.engagement_metrics")
    setattr(
        fake_engagement_module, "EngagementMonitor", type("EngagementMonitor", (), {})
    )

    monkeypatch.setitem(
        sys.modules, "src.integrations.youtube_client", fake_youtube_module
    )
    monkeypatch.setitem(sys.modules, "src.integrations.ai_client", fake_ai_module)
    monkeypatch.setitem(
        sys.modules,
        "src.processing.enhanced_thumbnail_generator",
        fake_thumbnail_module,
    )
    monkeypatch.setitem(
        sys.modules, "src.monitoring.engagement_metrics", fake_engagement_module
    )

    sys.modules.pop("src.management.channel_manager", None)
    channel_manager_module = importlib.import_module("src.management.channel_manager")
    ChannelManager = channel_manager_module.ChannelManager

    manager = ChannelManager.__new__(ChannelManager)
    manager.logger = logging.getLogger(__name__)
    manager._get_db_manager = lambda: db_manager

    resolved_path = manager._get_video_path("yt_video_124")
    assert resolved_path is None
