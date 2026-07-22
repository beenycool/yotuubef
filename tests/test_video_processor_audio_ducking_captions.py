from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.models import NarrativeSegment, AudioDuckingConfig, VideoAnalysis
from src.processing.video_processor import VideoProcessor
from src.hybrid_documentary_state_machine import generate_dynamic_captions


def test_video_processor_initializes_audio_processor_and_caption_generator():
    processor = VideoProcessor()
    assert hasattr(processor, "audio_processor")
    assert hasattr(processor, "caption_generator")


def test_video_processor_apply_audio_ducking_delegates_to_advanced_audio_processor():
    processor = VideoProcessor()
    mock_music = MagicMock()
    mock_segments = [
        NarrativeSegment(
            text="Hello world",
            time_seconds=1.0,
            intended_duration_seconds=3.0,
            narration="Hello world",
        )
    ]
    ducking_config = AudioDuckingConfig(duck_volume=0.2)

    with patch.object(
        processor.audio_processor, "process_audio_with_ducking", return_value=mock_music
    ) as mock_ducking:
        result = processor._apply_audio_ducking(
            mock_music, mock_segments, ducking_config=ducking_config
        )
        assert result == mock_music
        mock_ducking.assert_called_once_with(
            background_music=mock_music,
            narrative_segments=mock_segments,
            ducking_config=ducking_config,
        )


def test_video_processor_add_captions_with_known_text():
    processor = VideoProcessor()
    mock_video = MagicMock()
    mock_captioned_video = MagicMock()
    mock_segments = [
        NarrativeSegment(
            text="Testing dynamic captions",
            time_seconds=0.0,
            intended_duration_seconds=2.0,
            narration="Testing dynamic captions",
        )
    ]

    with patch.object(
        processor.caption_generator,
        "generate_captions_from_known_text",
        return_value=mock_captioned_video,
    ) as mock_gen:
        res = processor.add_captions(mock_video, narrative_segments=mock_segments)
        assert res == mock_captioned_video
        mock_gen.assert_called_once_with(mock_video, mock_segments)


def test_hybrid_state_machine_generate_dynamic_captions():
    mock_video = MagicMock()
    mock_captioned_video = MagicMock()
    segments = [
        NarrativeSegment(
            text="State machine captions",
            time_seconds=0.0,
            intended_duration_seconds=2.0,
            narration="State machine captions",
        )
    ]

    with patch(
        "src.processing.caption_generator.CaptionGenerator.generate_captions_from_known_text",
        return_value=mock_captioned_video,
    ):
        result = generate_dynamic_captions(mock_video, segments=segments)
        assert result == mock_captioned_video
