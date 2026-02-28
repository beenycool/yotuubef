import logging
import importlib
import sys
import types
import asyncio
from datetime import datetime, timedelta

import pytest

from src.integrations.youtube_client import YouTubeClient
from src.monitoring.engagement_metrics import EngagementMetricsDB


class _FakeAnalyticsReports:
    def query(self, **kwargs):
        return kwargs


class _FakeAnalyticsService:
    def reports(self):
        return _FakeAnalyticsReports()


def test_get_video_analytics_parses_analytics_report_rows(monkeypatch):
    client = YouTubeClient()
    client.analytics_service = _FakeAnalyticsService()

    async def _noop_init():
        return None

    async def _fake_video_info(video_id):
        return {
            "id": video_id,
            "statistics": {
                "viewCount": "100",
                "likeCount": "10",
                "commentCount": "2",
            },
        }

    async def _fake_execute(_request):
        return {
            "columnHeaders": [
                {"name": "impressionsCtr", "columnType": "METRIC"},
                {"name": "averageViewPercentage", "columnType": "METRIC"},
                {"name": "likes", "columnType": "METRIC"},
                {"name": "views", "columnType": "METRIC"},
                {"name": "comments", "columnType": "METRIC"},
                {"name": "averageViewDuration", "columnType": "METRIC"},
                {"name": "impressions", "columnType": "METRIC"},
            ],
            "rows": [["5.5", "62.5", "20", "300", "11", "47", "2000"]],
        }

    monkeypatch.setattr(client, "_ensure_services_initialized", _noop_init)
    monkeypatch.setattr(client, "get_video_info", _fake_video_info)
    monkeypatch.setattr(client, "_execute_request_async", _fake_execute)

    analytics = asyncio.run(client.get_video_analytics("vid123"))

    assert analytics is not None
    assert analytics["impressions"] == 2000
    assert analytics["ctr"] == pytest.approx(0.055)
    assert analytics["clicks"] == 110
    assert analytics["average_view_duration"] == pytest.approx(47.0)
    assert analytics["average_view_percentage"] == pytest.approx(62.5)
    assert analytics["likes"] == 20
    assert analytics["comments"] == 11


def test_get_video_analytics_falls_back_when_analytics_unavailable(monkeypatch):
    client = YouTubeClient()
    client.analytics_service = None

    async def _noop_init():
        return None

    async def _fake_video_info(video_id):
        return {
            "id": video_id,
            "statistics": {
                "viewCount": "1000",
                "likeCount": "50",
                "commentCount": "8",
                "dislikeCount": "1",
            },
        }

    monkeypatch.setattr(client, "_ensure_services_initialized", _noop_init)
    monkeypatch.setattr(client, "get_video_info", _fake_video_info)

    analytics = asyncio.run(client.get_video_analytics("vid123"))

    assert analytics is not None
    assert analytics["views"] == 1000
    assert analytics["likes"] == 50
    assert analytics["comments"] == 8
    assert analytics["impressions"] == 0
    assert analytics["ctr"] == 0.0
    assert analytics["average_view_duration"] == 0.0
    assert analytics["average_view_percentage"] == 0.0


def test_performance_snapshot_upsert_replaces_same_video_same_day(tmp_path):
    db = EngagementMetricsDB(db_path=tmp_path / "engagement_metrics.db")

    snapshot_date = "2026-02-28"
    ok_first = db.upsert_performance_snapshot(
        video_id="v1",
        snapshot_date=snapshot_date,
        analytics_data={"views": 100, "impressions": 1000, "ctr": 0.04},
    )
    ok_second = db.upsert_performance_snapshot(
        video_id="v1",
        snapshot_date=snapshot_date,
        analytics_data={"views": 120, "impressions": 1200, "ctr": 0.08},
    )

    rows = db.get_recent_performance_snapshots("v1", limit=5)

    assert ok_first is True
    assert ok_second is True
    assert len(rows) == 1
    assert rows[0]["views"] == 120
    assert rows[0]["impressions"] == 1200
    assert rows[0]["ctr"] == pytest.approx(0.08)


def test_winner_selection_threshold_gating_and_winner_choice(monkeypatch):
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
        sys.modules,
        "src.monitoring.engagement_metrics",
        fake_engagement_module,
    )

    sys.modules.pop("src.management.channel_manager", None)
    channel_manager_module = importlib.import_module("src.management.channel_manager")
    ChannelManager = channel_manager_module.ChannelManager

    manager = ChannelManager.__new__(ChannelManager)
    manager.logger = logging.getLogger(__name__)
    manager.thumbnail_test_duration = 24
    manager.min_impressions_for_test = 1000
    manager.min_variant_samples = 1

    start_time = (datetime.now() - timedelta(hours=26)).isoformat()
    test_info = {
        "start_time": start_time,
        "variants": [{"variant_id": "A"}, {"variant_id": "B"}],
        "performance_data": [
            {"variant_index": 0, "stats": {"impressions": 500, "ctr": 0.03}},
            {"variant_index": 1, "stats": {"impressions": 500, "ctr": 0.06}},
            {"variant_index": 0, "stats": {"impressions": 700, "ctr": 0.04}},
            {"variant_index": 1, "stats": {"impressions": 700, "ctr": 0.05}},
        ],
    }

    ready, reason = manager._evaluate_thumbnail_test_readiness(
        test_info,
        current_stats={"impressions": 900},
    )
    assert ready is False
    assert "impression threshold not met" in reason

    analysis = manager._analyze_thumbnail_test_results(test_info)
    assert analysis is not None
    assert analysis["winner_index"] == 1
    assert analysis["variant_ctrs"][1] > analysis["variant_ctrs"][0]
