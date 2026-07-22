"""
Unit tests verifying the integration of AssetVectorStore and CinematicEditor:
1. AssetVectorStore.get_similar_assets integrated with BraveImageClient / image search.
2. CinematicEditor integrated into EnhancedVideoOrchestrator for scene analysis and pacing adjustment.
"""

import pytest
import asyncio
from pathlib import Path
from types import SimpleNamespace

from src.processing.asset_vector_store import AssetVectorStore
from src.processing.image_search_client import BraveImageClient
from src.processing.cinematic_editor import CinematicEditor
from src.enhanced_orchestrator import EnhancedVideoOrchestrator
from src.models import VideoAnalysisEnhanced


@pytest.mark.asyncio
async def test_asset_vector_store_get_similar_assets(tmp_path):
    store = AssetVectorStore(persist_dir=tmp_path / "vectors")
    await store.initialize()

    # Create dummy media files
    img_dir = tmp_path / "broll_images"
    img_dir.mkdir(parents=True)
    img_file = img_dir / "mountain_landscape.jpg"
    img_file.write_bytes(b"fake image content")

    count = await store.scan_directory(img_dir)
    assert count == 1

    # Query using text string query via get_similar_assets
    results = await store.get_similar_assets("mountain landscape", top_k=3)
    assert len(results) >= 1
    assert "mountain" in results[0]["tags"] or "landscape" in results[0]["tags"]

    # Query using existing file path via get_similar_assets
    results_path = await store.get_similar_assets(str(img_file), top_k=3)
    assert len(results_path) >= 1


@pytest.mark.asyncio
async def test_image_search_client_uses_local_vector_store_before_web_search(tmp_path):
    store = AssetVectorStore(persist_dir=tmp_path / "vectors")
    await store.initialize()

    img_dir = tmp_path / "broll_images"
    img_dir.mkdir(parents=True)
    img_file = img_dir / "sports_car_race.jpg"
    img_file.write_bytes(b"fake image data")

    await store.scan_directory(img_dir)

    client = BraveImageClient(asset_store=store)

    # Search for images - should hit local vector store first
    results = await client.search_images("sports car race", count=1)
    assert len(results) == 1
    assert results[0]["type"] == "local_vector_store"
    assert results[0]["source"] == "local_cache"

    # Download image - should resolve file:// URL directly without external network call
    downloaded = await client.download_image(results[0]["url"], "sports car race")
    assert downloaded is not None
    assert downloaded.exists()


@pytest.mark.asyncio
async def test_cinematic_editor_pacing_adjustment():
    editor = CinematicEditor()
    segments = [
        {"narration": "Short text", "emotion": "neutral"},
        {
            "narration": "This is a very long segment narration with a massive amount of words intended to speed up pacing dynamically for fast action.",
            "emotion": "excited",
        },
        {"narration": "A dramatic quiet pause", "emotion": "dramatic"},
    ]

    adjusted = editor.adjust_pacing(segments, target_style="dynamic")
    assert len(adjusted) == 3
    assert adjusted[0]["pacing"] == "normal"
    assert adjusted[1]["pacing"] == "fast"
    assert adjusted[2]["pacing"] == "slow"


@pytest.mark.asyncio
async def test_enhanced_orchestrator_cinematic_editor_integration(tmp_path):
    orchestrator = EnhancedVideoOrchestrator()
    assert hasattr(orchestrator, "cinematic_editor")
    assert isinstance(orchestrator.cinematic_editor, CinematicEditor)

    # Test pacing adjustment method
    segments = [
        {"narration": "Quick fast scene with action!", "emotion": "excited"},
        {"narration": "Calm reflection.", "emotion": "calm"},
    ]
    adjusted = orchestrator.adjust_cinematic_pacing(segments, target_style="dynamic")
    assert adjusted[0]["pacing"] in ["fast", "normal"]
    assert adjusted[1]["pacing"] == "slow"

    # Test analysis stub
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"fake mp4 content")

    # analyze_rendered_video_cinematically should safely return VideoAnalysisEnhanced instance
    analysis = orchestrator.analyze_rendered_video_cinematically(dummy_video)
    assert isinstance(analysis, VideoAnalysisEnhanced)
