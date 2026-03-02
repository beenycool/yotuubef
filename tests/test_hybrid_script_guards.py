import logging
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.enhanced_orchestrator import EnhancedVideoOrchestrator
import src.enhanced_orchestrator as orchestrator_module


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


def test_merge_downloaded_media_entries_dedupes_by_source_url():
    orchestrator = _build_orchestrator_stub()

    existing = [
        {
            "media_type": "image",
            "source_url": "https://Example.com/proof.png#fragment",
            "local_path": "research/media_images/image_001.png",
        }
    ]
    new = [
        {
            "media_type": "image",
            "source_url": "https://example.com/proof.png",
            "local_path": "research/media_images/image_008.png",
        },
        {
            "media_type": "image",
            "source_url": "https://example.com/another.png",
            "local_path": "research/media_images/image_009.png",
        },
    ]

    merged = orchestrator._merge_downloaded_media_entries(existing, new)
    local_paths = [item.get("local_path") for item in merged]

    assert len(merged) == 2
    assert "research/media_images/image_001.png" in local_paths
    assert "research/media_images/image_008.png" not in local_paths
    assert "research/media_images/image_009.png" in local_paths


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
async def test_apply_hybrid_image_relevance_mapping_counts_unique_image_sources(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HYBRID_IMAGE_RELEVANCE_MIN_SCORE", "70")
    monkeypatch.setenv("HYBRID_IMAGE_RELEVANCE_TOP_K", "3")
    monkeypatch.setenv("HYBRID_IMAGE_RELEVANCE_MAX_CALLS", "10")

    project_dir = tmp_path / "run3"
    (project_dir / "research" / "media_images").mkdir(parents=True)
    (project_dir / "research" / "evidence").mkdir(parents=True)

    img_a = project_dir / "research" / "media_images" / "image_001.jpg"
    img_b = project_dir / "research" / "media_images" / "image_002.jpg"
    img_a.write_bytes(b"a")
    img_b.write_bytes(b"b")

    evidence_index_path = project_dir / "research" / "evidence" / "evidence_index.json"
    evidence_index_path.write_text(
        json.dumps(
            {
                "downloaded_media": [
                    {
                        "media_type": "image",
                        "local_path": "research/media_images/image_001.jpg",
                        "query": "proof query",
                        "title": "Proof A",
                        "source_url": "https://example.com/same-proof.jpg",
                    },
                    {
                        "media_type": "image",
                        "local_path": "research/media_images/image_002.jpg",
                        "query": "proof query",
                        "title": "Proof B",
                        "source_url": "https://example.com/same-proof.jpg",
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

    async def _always_high(media_path: Path, segment, media_meta, *, project_dir=None):
        _ = media_path, segment, media_meta, project_dir
        return {"score": 95, "relevant": True, "reason": "same source duplicate"}

    orchestrator._score_hybrid_visual_relevance = _always_high

    payload = {
        "phase": "SCRIPTING",
        "segments": [
            {
                "time_seconds": 0.0,
                "intended_duration_seconds": 6.0,
                "narration": "Show proof image.",
                "visual_directive": "Use the best screenshot.",
                "visual_asset_path": "",
            }
        ],
    }
    assets = [
        "research/media_images/image_001.jpg",
        "research/media_images/image_002.jpg",
    ]

    await orchestrator._apply_hybrid_image_relevance_mapping(state, payload, assets)

    report_path = project_dir / "research" / "evidence" / "media_relevance_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["segments"][0]["approved_count_images"] == 1


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


@pytest.mark.asyncio
async def test_retry_loop_downloads_more_images_and_updates_evidence_index(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HYBRID_IMAGE_RELEVANCE_MIN_APPROVED_PER_SEGMENT", "3")
    monkeypatch.setenv("HYBRID_IMAGE_RELEVANCE_MAX_RETRIES", "2")

    project_dir = tmp_path / "retry1"
    (project_dir / "research" / "evidence").mkdir(parents=True)

    evidence_index_path = project_dir / "research" / "evidence" / "evidence_index.json"
    evidence_index_path.write_text(
        json.dumps({"downloaded_media": [], "image_queries": ["existing query"]}),
        encoding="utf-8",
    )

    state = SimpleNamespace(
        project_dir=str(project_dir),
        metadata={"evidence_index_path": str(evidence_index_path)},
    )

    orchestrator = _build_orchestrator_stub()

    # save_run_state expects a RunState (pydantic). Stub it for this unit test.
    monkeypatch.setattr(orchestrator_module, "save_run_state", lambda _state: None)

    calls = {"apply": 0, "search": 0, "download": 0}

    async def _fake_apply(state_arg, script_payload, assets):
        _ = state_arg, script_payload, assets
        calls["apply"] += 1
        if calls["apply"] == 1:
            return {"segments": [{"segment_index": 0, "approved_count_images": 0}]}
        return {"segments": [{"segment_index": 0, "approved_count_images": 3}]}

    async def _fake_queries(
        state_arg,
        script_payload,
        *,
        segment_indexes,
        existing_queries=None,
        max_per_segment=3,
    ):
        _ = (
            state_arg,
            script_payload,
            segment_indexes,
            existing_queries,
            max_per_segment,
        )
        return ["new evidence query"]

    async def _fake_search(*, image_queries, video_queries, count):
        _ = image_queries, video_queries, count
        calls["search"] += 1
        return {
            "image": {"new evidence query": [{"url": "https://example.com/a.jpg"}]},
            "video": {},
            "web": {},
        }

    async def _fake_download(state_arg, search_payload):
        _ = state_arg, search_payload
        calls["download"] += 1
        return [
            {
                "media_type": "image",
                "source_url": "https://example.com/a.jpg",
                "query": "new evidence query",
                "title": "A",
                "local_path": "research/media_images/image_999.jpg",
            }
        ]

    orchestrator._apply_hybrid_image_relevance_mapping = _fake_apply
    orchestrator._generate_supplementary_image_queries = _fake_queries
    orchestrator._run_hybrid_media_queries = _fake_search
    orchestrator._download_hybrid_media_assets = _fake_download

    script_payload = {
        "phase": "SCRIPTING",
        "segments": [
            {
                "time_seconds": 0.0,
                "intended_duration_seconds": 6.0,
                "narration": "Show the specific leaderboard removal screenshot.",
                "visual_directive": "Zoom into receipts and dates.",
                "visual_asset_path": "",
                "evidence_refs": [],
            }
        ],
        "title": "",
        "hook": "",
        "loop_bridge": "",
        "sources_to_check": [],
        "hashtags": [],
    }

    report = await orchestrator._ensure_min_approved_images_per_segment(
        state,
        script_payload,
    )

    assert calls["apply"] >= 2
    assert calls["search"] == 1
    assert calls["download"] == 1
    assert report["segments"][0]["approved_count_images"] == 3

    updated = json.loads(evidence_index_path.read_text(encoding="utf-8"))
    assert any(
        item.get("local_path") == "research/media_images/image_999.jpg"
        for item in updated.get("downloaded_media", [])
    )
    assert "new evidence query" in updated.get("supplementary_image_queries", [])
    retry_results = updated.get("supplementary_search_results_paths", [])
    assert retry_results and "media_search_results_retry_1.json" in retry_results[0]


@pytest.mark.asyncio
async def test_retry_loop_stops_after_max_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("HYBRID_IMAGE_RELEVANCE_MIN_APPROVED_PER_SEGMENT", "3")
    monkeypatch.setenv("HYBRID_IMAGE_RELEVANCE_MAX_RETRIES", "0")

    project_dir = tmp_path / "retry2"
    (project_dir / "research" / "evidence").mkdir(parents=True)

    evidence_index_path = project_dir / "research" / "evidence" / "evidence_index.json"
    evidence_index_path.write_text(
        json.dumps({"downloaded_media": []}), encoding="utf-8"
    )

    state = SimpleNamespace(
        project_dir=str(project_dir),
        metadata={"evidence_index_path": str(evidence_index_path)},
    )
    orchestrator = _build_orchestrator_stub()
    monkeypatch.setattr(orchestrator_module, "save_run_state", lambda _state: None)

    async def _always_under(state_arg, script_payload, assets):
        _ = state_arg, script_payload, assets
        return {"segments": [{"segment_index": 0, "approved_count_images": 0}]}

    orchestrator._apply_hybrid_image_relevance_mapping = _always_under

    script_payload = {
        "phase": "SCRIPTING",
        "segments": [
            {
                "time_seconds": 0.0,
                "intended_duration_seconds": 6.0,
                "narration": "Beat",
                "visual_directive": "Directive",
                "visual_asset_path": "",
            }
        ],
        "title": "",
        "hook": "",
        "loop_bridge": "",
        "sources_to_check": [],
        "hashtags": [],
    }

    report = await orchestrator._ensure_min_approved_images_per_segment(
        state, script_payload
    )
    assert report["segments"][0]["approved_count_images"] == 0


@pytest.mark.asyncio
async def test_agentic_visual_loop_requests_images_and_revises_script(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HYBRID_VISUAL_FEEDBACK_MAX_ROUNDS", "2")
    monkeypatch.setenv("HYBRID_VISUAL_REVISION_ENABLED", "true")

    project_dir = tmp_path / "agentic1"
    (project_dir / "research" / "evidence").mkdir(parents=True)

    evidence_index_path = project_dir / "research" / "evidence" / "evidence_index.json"
    evidence_index_path.write_text(
        json.dumps({"downloaded_media": []}),
        encoding="utf-8",
    )

    state = SimpleNamespace(
        project_dir=str(project_dir),
        metadata={"evidence_index_path": str(evidence_index_path)},
    )

    orchestrator = _build_orchestrator_stub()
    monkeypatch.setattr(orchestrator_module, "save_run_state", lambda _state: None)

    script_payload = {
        "phase": "SCRIPTING",
        "segments": [
            {
                "time_seconds": 0.0,
                "intended_duration_seconds": 6.0,
                "narration": "Original narration about the dispute.",
                "visual_directive": "Show the key screenshot with dates.",
                "visual_asset_path": "",
            }
        ],
        "title": "",
        "hook": "",
        "loop_bridge": "",
        "sources_to_check": [],
        "hashtags": [],
    }

    async def _fake_generate(state_arg, phase_arg, context):
        _ = state_arg, phase_arg, context
        return script_payload

    orchestrator._generate_hybrid_phase_payload = _fake_generate

    def _fake_catalog(state_arg):
        _ = state_arg
        return {
            "images": [
                {
                    "id": 0,
                    "local_path": "research/media_images/image_aaa.jpg",
                    "query": "existing evidence",
                    "title": "Screenshot",
                    "source_url": "https://example.com/screenshot",
                }
            ]
        }

    orchestrator._build_visual_catalog_for_script = _fake_catalog

    async def _fake_feedback(state_arg, script_arg, visual_catalog):
        _ = state_arg, script_arg, visual_catalog
        return {
            "segments": [
                {
                    "segment_index": 0,
                    "status": "needs_images",
                    "preferred_assets": ["research/media_images/image_aaa.jpg"],
                    "new_image_queries": [
                        "q1 evidence",
                        "q1 evidence",
                    ],  # duplicate to test dedupe
                    "revised_narration": "Revised narration with the same facts.",
                }
            ]
        }

    orchestrator._run_visual_feedback_agent = _fake_feedback

    calls = {"search": 0, "download": 0, "ensure": 0}

    async def _fake_search(*, image_queries, video_queries, count):
        _ = video_queries, count
        calls["search"] += 1
        assert image_queries == ["q1 evidence"]
        return {
            "image": {"q1 evidence": [{"url": "https://example.com/new.jpg"}]},
            "video": {},
            "web": {},
        }

    async def _fake_download(state_arg, search_payload):
        _ = state_arg, search_payload
        calls["download"] += 1
        return [
            {
                "media_type": "image",
                "source_url": "https://example.com/new.jpg",
                "query": "q1 evidence",
                "title": "New",
                "local_path": "research/media_images/image_123.jpg",
            }
        ]

    async def _fake_ensure(state_arg, script_arg):
        _ = state_arg, script_arg
        calls["ensure"] += 1
        return {"segments": [{"segment_index": 0, "approved_count_images": 3}]}

    orchestrator._run_hybrid_media_queries = _fake_search
    orchestrator._download_hybrid_media_assets = _fake_download
    orchestrator._ensure_min_approved_images_per_segment = _fake_ensure

    result = await orchestrator._run_agentic_scripting_with_visual_loop(
        state,
        "SCRIPTING",
        "context",
    )

    segment = result["segments"][0]
    assert segment["narration"].startswith("Revised narration")
    assert segment["visual_asset_path"] == "research/media_images/image_aaa.jpg"
    assert calls["search"] == 1
    assert calls["download"] == 1
    assert calls["ensure"] == 1

    updated = json.loads(evidence_index_path.read_text(encoding="utf-8"))
    assert any(
        item.get("local_path") == "research/media_images/image_123.jpg"
        for item in updated.get("downloaded_media", [])
    )
    assert "q1 evidence" in updated.get("supplementary_image_queries", [])


@pytest.mark.asyncio
async def test_agentic_visual_loop_stops_when_no_images(tmp_path, monkeypatch):
    monkeypatch.setenv("HYBRID_VISUAL_FEEDBACK_MAX_ROUNDS", "2")

    project_dir = tmp_path / "agentic2"
    (project_dir / "research" / "evidence").mkdir(parents=True)

    evidence_index_path = project_dir / "research" / "evidence" / "evidence_index.json"
    evidence_index_path.write_text(
        json.dumps({"downloaded_media": []}), encoding="utf-8"
    )

    state = SimpleNamespace(
        project_dir=str(project_dir),
        metadata={"evidence_index_path": str(evidence_index_path)},
    )

    orchestrator = _build_orchestrator_stub()
    monkeypatch.setattr(orchestrator_module, "save_run_state", lambda _state: None)

    script_payload = {
        "phase": "SCRIPTING",
        "segments": [
            {
                "time_seconds": 0.0,
                "intended_duration_seconds": 6.0,
                "narration": "Original.",
                "visual_directive": "Directive.",
                "visual_asset_path": "",
            }
        ],
        "title": "",
        "hook": "",
        "loop_bridge": "",
        "sources_to_check": [],
        "hashtags": [],
    }

    async def _fake_generate(state_arg, phase_arg, context):
        _ = state_arg, phase_arg, context
        return script_payload

    orchestrator._generate_hybrid_phase_payload = _fake_generate

    def _empty_catalog(state_arg):
        _ = state_arg
        return {"images": []}

    orchestrator._build_visual_catalog_for_script = _empty_catalog

    calls = {"ensure": 0}

    async def _fake_ensure(state_arg, script_arg):
        _ = state_arg, script_arg
        calls["ensure"] += 1
        return {"segments": [{"segment_index": 0, "approved_count_images": 1}]}

    orchestrator._ensure_min_approved_images_per_segment = _fake_ensure

    result = await orchestrator._run_agentic_scripting_with_visual_loop(
        state,
        "SCRIPTING",
        "context",
    )

    assert result["segments"][0]["narration"] == "Original."
    assert calls["ensure"] == 1
