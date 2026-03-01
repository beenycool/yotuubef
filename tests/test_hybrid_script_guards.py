import logging
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.enhanced_orchestrator import EnhancedVideoOrchestrator


def _build_orchestrator_stub() -> EnhancedVideoOrchestrator:
    orchestrator = EnhancedVideoOrchestrator.__new__(EnhancedVideoOrchestrator)
    orchestrator.logger = logging.getLogger(__name__)
    return orchestrator


def test_build_gemini_research_query_includes_topic():
    query = EnhancedVideoOrchestrator._build_gemini_research_query(
        "Todd Rogers Dragster",
        "Search for primary sources and archived links.",
    )
    assert "todd rogers dragster" in query.lower()


def test_prepare_idea_generation_payload_creates_specific_deep_research_prompt():
    orchestrator = _build_orchestrator_stub()
    payload = {
        "phase": "IDEA_GENERATION",
        "angles": [
            {
                "id": "A1",
                "title": "Dream Speedrun Disqualification",
                "hook": "How did impossible RNG get accepted for months?",
                "viability_score": 90,
                "source_urls": [
                    "https://www.speedrun.com/mc/forums/wh21i",
                    "https://mcspeedrun.com/dream.pdf",
                ],
            }
        ],
        "gemini_deep_research_prompt": "Search for sources",
        "next_phase": "WAIT_FOR_GEMINI_REPORT",
    }

    prepared = orchestrator._prepare_idea_generation_payload(payload, "run1")
    prompt = prepared["gemini_deep_research_prompt"]

    assert "dream speedrun disqualification" in prompt.lower()
    assert "how did impossible rng" in prompt.lower()
    assert "https://www.speedrun.com/mc/forums/wh21i" in prompt
    assert "https://mcspeedrun.com/dream.pdf" in prompt
    assert "Chronology" in prompt


def test_build_gemini_research_query_uses_seed_source_hosts():
    query = EnhancedVideoOrchestrator._build_gemini_research_query(
        "Dream Speedrun Disqualification",
        """Topic: Dream Speedrun Disqualification
Seed sources to verify:
- https://www.speedrun.com/mc/forums/wh21i
- https://mcspeedrun.com/dream.pdf
""",
    )

    lowered = query.lower()
    assert "dream speedrun disqualification" in lowered
    assert "site:speedrun.com" in lowered
    assert "site:mcspeedrun.com" in lowered


def test_research_relevance_guard_filters_generic_text():
    orchestrator = _build_orchestrator_stub()
    report_text = (
        "EXTERNAL FACTS: Library and court records overview with no game references."
    )
    angles = [{"source_urls": ["https://kotaku.com/example"]}]

    is_relevant = orchestrator._is_research_report_relevant(
        report_text,
        "Todd Rogers Dragster controversy",
        angles,
    )

    assert is_relevant is False


def test_enforce_script_asset_mapping_uses_visual_assets_and_balances_duration(
    monkeypatch,
):
    monkeypatch.delenv("HYBRID_TARGET_DURATION_SECONDS", raising=False)
    orchestrator = _build_orchestrator_stub()
    payload = {
        "phase": "SCRIPTING",
        "segments": [
            {
                "time_seconds": 0.0,
                "intended_duration_seconds": 2.5,
                "narration": "Hook",
                "visual_asset_path": "research/evidence/evidence_index.json",
            },
            {
                "time_seconds": 2.5,
                "intended_duration_seconds": 3.0,
                "narration": "Proof beat",
                "visual_asset_path": "research/evidence/media_search_results.json",
            },
            {
                "time_seconds": 5.5,
                "intended_duration_seconds": 33.0,
                "narration": "To this day, the 5.57 second barrier stands as the true human limit.",
                "visual_asset_path": "research/evidence/evidence_index.json",
            },
        ],
    }
    assets = [
        "research/evidence/evidence_index.json",
        "research/media_images/image_001.jpg",
        "raw_media/video_001.mp4",
    ]

    orchestrator._enforce_script_asset_mapping(payload, assets)

    segments = payload["segments"]
    assert segments[0]["visual_asset_path"].endswith(".jpg") or segments[0][
        "visual_asset_path"
    ].endswith(".mp4")
    assert all(float(seg["intended_duration_seconds"]) <= 14.0 for seg in segments)

    total_duration = sum(float(seg["intended_duration_seconds"]) for seg in segments)
    assert round(total_duration, 2) == 45.0

    assert float(segments[0]["time_seconds"]) == 0.0
    assert float(segments[1]["time_seconds"]) > float(segments[0]["time_seconds"])
    assert float(segments[2]["time_seconds"]) > float(segments[1]["time_seconds"])


def test_enforce_script_asset_mapping_honors_target_duration_override(monkeypatch):
    monkeypatch.setenv("HYBRID_TARGET_DURATION_SECONDS", "60")
    orchestrator = _build_orchestrator_stub()
    payload = {
        "phase": "SCRIPTING",
        "segments": [
            {
                "time_seconds": 0.0,
                "intended_duration_seconds": 5.0,
                "narration": "Beat one with concrete source and date.",
                "visual_asset_path": "",
            },
            {
                "time_seconds": 5.0,
                "intended_duration_seconds": 5.0,
                "narration": "Beat two with another receipt and quote.",
                "visual_asset_path": "",
            },
            {
                "time_seconds": 10.0,
                "intended_duration_seconds": 5.0,
                "narration": "Beat three with timeline fallout details.",
                "visual_asset_path": "",
            },
            {
                "time_seconds": 15.0,
                "intended_duration_seconds": 5.0,
                "narration": "Beat four loop bridge.",
                "visual_asset_path": "",
            },
            {
                "time_seconds": 20.0,
                "intended_duration_seconds": 5.0,
                "narration": "Beat five CTA.",
                "visual_asset_path": "",
            },
        ],
    }
    assets = ["research/media_images/image_001.jpg"]

    orchestrator._enforce_script_asset_mapping(payload, assets)

    total_duration = sum(
        float(segment["intended_duration_seconds"]) for segment in payload["segments"]
    )
    assert round(total_duration, 2) == 60.0


@pytest.mark.asyncio
async def test_apply_hybrid_image_relevance_mapping_prefers_high_score_candidate(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HYBRID_IMAGE_RELEVANCE_MIN_SCORE", "70")
    monkeypatch.setenv("HYBRID_IMAGE_RELEVANCE_TOP_K", "2")
    monkeypatch.setenv("HYBRID_IMAGE_RELEVANCE_MAX_CALLS", "10")

    project_dir = tmp_path / "run1"
    (project_dir / "research" / "media_images").mkdir(parents=True)
    (project_dir / "research" / "evidence").mkdir(parents=True)
    (project_dir / "raw_media").mkdir(parents=True)

    img_relevant = project_dir / "research" / "media_images" / "image_001.jpg"
    img_irrelevant = project_dir / "research" / "media_images" / "image_007.jpg"
    video_fallback = project_dir / "raw_media" / "video_001.mp4"
    img_relevant.write_bytes(b"relevant")
    img_irrelevant.write_bytes(b"irrelevant")
    video_fallback.write_bytes(b"video")

    evidence_index_path = project_dir / "research" / "evidence" / "evidence_index.json"
    evidence_index_path.write_text(
        json.dumps(
            {
                "downloaded_media": [
                    {
                        "media_type": "image",
                        "local_path": "research/media_images/image_001.jpg",
                        "query": "dream speedrun leaderboard removed",
                        "title": "Dream leaderboard screenshot",
                        "source_url": "https://example.com/relevant",
                    },
                    {
                        "media_type": "image",
                        "local_path": "research/media_images/image_007.jpg",
                        "query": "ender pearl guide",
                        "title": "How to farm pearls",
                        "source_url": "https://example.com/irrelevant",
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    state = SimpleNamespace(
        project_dir=str(project_dir),
        metadata={"evidence_index_path": str(evidence_index_path)},
    )
    orchestrator = _build_orchestrator_stub()

    async def _fake_score(media_path: Path, segment, media_meta, *, project_dir=None):
        _ = segment, media_meta, project_dir
        if media_path.name == "image_001.jpg":
            return {"score": 92, "relevant": True, "reason": "Exact match"}
        return {"score": 28, "relevant": False, "reason": "Off-topic"}

    orchestrator._score_hybrid_visual_relevance = _fake_score

    payload = {
        "phase": "SCRIPTING",
        "segments": [
            {
                "time_seconds": 0.0,
                "intended_duration_seconds": 6.0,
                "narration": "Show the leaderboard where Dream's run was removed.",
                "visual_directive": "Zoom into leaderboard receipts.",
                "visual_asset_path": "research/media_images/image_007.jpg",
            }
        ],
    }
    assets = [
        "research/media_images/image_007.jpg",
        "research/media_images/image_001.jpg",
        "raw_media/video_001.mp4",
    ]

    await orchestrator._apply_hybrid_image_relevance_mapping(state, payload, assets)

    assert (
        payload["segments"][0]["visual_asset_path"]
        == "research/media_images/image_001.jpg"
    )

    report_path = project_dir / "research" / "evidence" / "media_relevance_report.json"
    assert report_path.exists()


@pytest.mark.asyncio
async def test_apply_hybrid_image_relevance_mapping_excludes_low_score_images(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HYBRID_IMAGE_RELEVANCE_MIN_SCORE", "70")

    project_dir = tmp_path / "run2"
    (project_dir / "research" / "media_images").mkdir(parents=True)
    (project_dir / "research" / "evidence").mkdir(parents=True)
    (project_dir / "raw_media").mkdir(parents=True)

    img_irrelevant = project_dir / "research" / "media_images" / "image_007.jpg"
    video_fallback = project_dir / "raw_media" / "video_001.mp4"
    img_irrelevant.write_bytes(b"irrelevant")
    video_fallback.write_bytes(b"video")

    evidence_index_path = project_dir / "research" / "evidence" / "evidence_index.json"
    evidence_index_path.write_text(
        json.dumps(
            {
                "downloaded_media": [
                    {
                        "media_type": "image",
                        "local_path": "research/media_images/image_007.jpg",
                        "query": "minecraft pearl farming guide",
                        "title": "general gameplay guide",
                        "source_url": "https://example.com/guide",
                    },
                    {
                        "media_type": "video",
                        "local_path": "raw_media/video_001.mp4",
                        "query": "dream speedrun evidence footage",
                        "title": "speedrun clip",
                        "source_url": "https://example.com/video",
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    state = SimpleNamespace(
        project_dir=str(project_dir),
        metadata={"evidence_index_path": str(evidence_index_path)},
    )
    orchestrator = _build_orchestrator_stub()

    async def _always_low_score(
        media_path: Path, segment, media_meta, *, project_dir=None
    ):
        _ = media_path, segment, media_meta, project_dir
        return {"score": 21, "relevant": False, "reason": "Not related"}

    orchestrator._score_hybrid_visual_relevance = _always_low_score

    payload = {
        "phase": "SCRIPTING",
        "segments": [
            {
                "time_seconds": 0.0,
                "intended_duration_seconds": 6.0,
                "narration": "Show Dream disqualification leaderboard proof.",
                "visual_directive": "Highlight removed entry.",
                "visual_asset_path": "research/media_images/image_007.jpg",
            }
        ],
    }
    assets = [
        "research/media_images/image_007.jpg",
        "raw_media/video_001.mp4",
    ]

    await orchestrator._apply_hybrid_image_relevance_mapping(state, payload, assets)

    assert payload["segments"][0]["visual_asset_path"] == ""


@pytest.mark.asyncio
async def test_apply_hybrid_image_relevance_mapping_can_select_relevant_video(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HYBRID_IMAGE_RELEVANCE_MIN_SCORE", "70")
    monkeypatch.setenv("HYBRID_IMAGE_RELEVANCE_TOP_K", "3")

    project_dir = tmp_path / "run3"
    (project_dir / "research" / "media_images").mkdir(parents=True)
    (project_dir / "research" / "evidence").mkdir(parents=True)
    (project_dir / "raw_media").mkdir(parents=True)

    img_irrelevant = project_dir / "research" / "media_images" / "image_007.jpg"
    video_relevant = project_dir / "raw_media" / "video_001.mp4"
    img_irrelevant.write_bytes(b"irrelevant")
    video_relevant.write_bytes(b"video")

    evidence_index_path = project_dir / "research" / "evidence" / "evidence_index.json"
    evidence_index_path.write_text(
        json.dumps(
            {
                "downloaded_media": [
                    {
                        "media_type": "image",
                        "local_path": "research/media_images/image_007.jpg",
                        "query": "general minecraft image",
                        "title": "ender pearl guide",
                        "source_url": "https://example.com/image",
                    },
                    {
                        "media_type": "video",
                        "local_path": "raw_media/video_001.mp4",
                        "query": "dream speedrun leaderboard dispute clip",
                        "title": "leaderboard removal evidence",
                        "source_url": "https://example.com/video",
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    state = SimpleNamespace(
        project_dir=str(project_dir),
        metadata={"evidence_index_path": str(evidence_index_path)},
    )
    orchestrator = _build_orchestrator_stub()

    async def _prefer_video(media_path: Path, segment, media_meta, *, project_dir=None):
        _ = segment, media_meta, project_dir
        if media_path.name == "video_001.mp4":
            return {"score": 90, "relevant": True, "reason": "Video evidence matches"}
        return {"score": 25, "relevant": False, "reason": "Image is generic"}

    orchestrator._score_hybrid_visual_relevance = _prefer_video

    payload = {
        "phase": "SCRIPTING",
        "segments": [
            {
                "time_seconds": 0.0,
                "intended_duration_seconds": 6.0,
                "narration": "Show the evidence clip of Dream leaderboard removal.",
                "visual_directive": "Use direct footage proving the dispute.",
                "visual_asset_path": "research/media_images/image_007.jpg",
            }
        ],
    }
    assets = [
        "research/media_images/image_007.jpg",
        "raw_media/video_001.mp4",
    ]

    await orchestrator._apply_hybrid_image_relevance_mapping(state, payload, assets)

    assert payload["segments"][0]["visual_asset_path"] == "raw_media/video_001.mp4"
