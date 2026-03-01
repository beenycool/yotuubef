import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.hybrid_documentary_state_machine import (
    PipelinePhase,
    load_run_state,
    set_phase,
    setup_project_workspace,
)
from src.integrations import tts_service as tts_module
from src.integrations.tts_service import TTSService
from src.models import EffectType, VisualCue
from src.processing.video_processor import VideoEffects, VideoProcessor
from src.enhanced_orchestrator import EnhancedVideoOrchestrator


def _load_phase_driver_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "phase_driver_module", root / "test.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load phase driver module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase_driver = _load_phase_driver_module()


def test_hybrid_workspace_state_persists_and_tracks_phase_transitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)

    project_dir = setup_project_workspace("Hybrid Phase Persistence")
    state = load_run_state(project_dir)
    initial_run_id = state.run_id

    assert state.current_phase == PipelinePhase.IDEA_GENERATION
    assert (project_dir / "raw_media").exists()

    set_phase(state, PipelinePhase.WAIT_FOR_GEMINI_REPORT, "idea complete")

    persisted = load_run_state(project_dir)
    assert persisted.current_phase == PipelinePhase.WAIT_FOR_GEMINI_REPORT
    assert len(persisted.transitions) == 1
    assert persisted.transitions[0].from_phase == PipelinePhase.IDEA_GENERATION
    assert persisted.transitions[0].to_phase == PipelinePhase.WAIT_FOR_GEMINI_REPORT

    setup_project_workspace("Hybrid Phase Persistence")
    persisted_again = load_run_state(project_dir)
    assert persisted_again.run_id == initial_run_id
    assert persisted_again.current_phase == PipelinePhase.WAIT_FOR_GEMINI_REPORT
    assert len(persisted_again.transitions) == 1


def test_phase_driver_require_keys_enforces_json_key_contract():
    with pytest.raises(ValueError, match="IDEA: missing keys"):
        phase_driver.require_keys(
            {"phase": "IDEA_GENERATION"}, ["phase", "angles"], "IDEA"
        )


def test_phase_driver_chat_json_invokes_validator_with_extracted_json():
    response = (
        "model preface\n"
        '{"phase":"IDEA_GENERATION","angles":[],"gemini_deep_research_prompt":"p","next_phase":"WAIT_FOR_GEMINI_REPORT"}'
        "\nmodel suffix"
    )
    validator = Mock()

    with patch.object(phase_driver, "chat_with_fallback", return_value=response):
        payload = phase_driver.chat_json("prompt", validator=validator, temperature=0.2)

    assert payload["phase"] == "IDEA_GENERATION"
    validator.assert_called_once_with(payload)


def test_tts_librosa_trim_is_non_destructive_when_no_trim_needed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service = TTSService.__new__(TTSService)

    audio = np.ones(100, dtype=np.float32)
    fake_librosa = SimpleNamespace(
        load=lambda *_args, **_kwargs: (audio, 1000),
        effects=SimpleNamespace(
            trim=lambda *_args, **_kwargs: (audio, np.array([0, 100]))
        ),
    )
    fake_sf = Mock()

    monkeypatch.setattr(tts_module, "LIBROSA_AVAILABLE", True)
    monkeypatch.setattr(tts_module, "SOUNDFILE_AVAILABLE", True)
    monkeypatch.setattr(tts_module, "_librosa", fake_librosa)
    monkeypatch.setattr(tts_module, "_sf", fake_sf)

    audio_path = tmp_path / "segment.wav"
    result = service._trim_silence_with_librosa(audio_path)

    assert result is False
    fake_sf.write.assert_not_called()


def test_perfect_loop_blend_guard_short_clips_are_noop():
    class DummyAudioClip:
        def __init__(self):
            self.duration = 0.2
            self.fps = 44100
            self.soundarray_calls = 0

        def to_soundarray(self, fps):
            self.soundarray_calls += 1
            return np.zeros((int(fps * self.duration), 1), dtype=np.float32)

    processor = VideoProcessor.__new__(VideoProcessor)
    processor.logger = logging.getLogger("test-video-processor")
    clip = DummyAudioClip()

    result = processor._blend_audio_loop_edges(clip, blend_seconds=0.5)  # type: ignore[arg-type]

    assert result is clip
    assert clip.soundarray_calls == 0


def test_callout_coordinate_parsing_from_directive_string():
    processor = VideoEffects.__new__(VideoEffects)
    processor.logger = logging.getLogger("test-video-processor")

    parsed = processor._parse_callout_coordinates(
        "highlight area x1=0.10 y1=0.20 x2=0.50 y2=0.80", (1000, 2000)
    )

    assert parsed == (100, 400, 500, 1600)


def test_visual_cue_accepts_visual_directive_payload():
    cue = VisualCue(
        timestamp_seconds=1.0,
        description="zoom into proof",
        effect_type=EffectType.HIGHLIGHT,
        visual_directive="x1=0.1 y1=0.2 x2=0.5 y2=0.8",
    )

    assert cue.visual_directive == "x1=0.1 y1=0.2 x2=0.5 y2=0.8"


def test_hybrid_script_asset_mapping_rewrites_unknown_paths():
    orchestrator = EnhancedVideoOrchestrator.__new__(EnhancedVideoOrchestrator)
    payload = {
        "segments": [
            {
                "visual_asset_path": "missing/path.png",
                "intended_duration_seconds": 5,
            }
        ]
    }

    orchestrator._enforce_script_asset_mapping(
        payload,
        assets=["research/media/images/image_001.jpg"],
    )

    assert payload["phase"] == "SCRIPTING"
    assert (
        payload["segments"][0]["visual_asset_path"]
        == "research/media/images/image_001.jpg"
    )
