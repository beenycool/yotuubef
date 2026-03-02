from unittest.mock import MagicMock
from src.utils.common_utils import (
    select_and_validate_segments,
    get_safe_filename,
    calculate_video_metrics,
    format_duration,
    validate_analysis_completeness
)

import pytest
from pathlib import Path
from src.utils.common_utils import validate_file_paths

def test_validate_file_paths_valid(tmp_path):
    input_file = tmp_path / "input.mp4"
    input_file.touch()

    output_dir = tmp_path / "output_dir"
    output_file = output_dir / "output.mp4"

    is_valid, msg = validate_file_paths(input_file, output_file)
    assert is_valid is True
    assert msg == ""
    assert output_dir.exists()

def test_validate_file_paths_valid_existing_output(tmp_path):
    input_file = tmp_path / "input.mp4"
    input_file.touch()

    output_file = tmp_path / "output.mp4"
    output_file.touch()

    is_valid, msg = validate_file_paths(input_file, output_file)
    assert is_valid is True
    assert msg == ""

def test_validate_file_paths_input_not_exists(tmp_path):
    input_file = tmp_path / "input.mp4"
    output_file = tmp_path / "output.mp4"

    is_valid, msg = validate_file_paths(input_file, output_file)
    assert is_valid is False
    assert "Input video file does not exist" in msg

def test_validate_file_paths_input_is_dir(tmp_path):
    input_dir = tmp_path / "input_dir"
    input_dir.mkdir()

    output_file = tmp_path / "output.mp4"

    is_valid, msg = validate_file_paths(input_dir, output_file)
    assert is_valid is False
    assert "Input path is not a file" in msg

def test_validate_file_paths_output_is_dir(tmp_path):
    input_file = tmp_path / "input.mp4"
    input_file.touch()

    output_dir = tmp_path / "output_dir"
    output_dir.mkdir()

    is_valid, msg = validate_file_paths(input_file, output_dir)
    assert is_valid is False
    assert "Output path exists but is not a file" in msg

def test_validate_file_paths_exception(tmp_path, monkeypatch):
    input_file = tmp_path / "input.mp4"
    input_file.touch()

    output_file = tmp_path / "output.mp4"

    def mock_exists(*args, **kwargs):
        raise RuntimeError("Mocked exception")

    monkeypatch.setattr(Path, "exists", mock_exists)

    is_valid, msg = validate_file_paths(input_file, output_file)
    assert is_valid is False
    assert "Path validation error: Mocked exception" in msg


def test_get_safe_filename():
    assert get_safe_filename("Normal Title") == "Normal_Title"
    assert get_safe_filename("Title with < > : \" / \\ | ? *") == "Title_with"
    assert get_safe_filename("A" * 60, max_length=50) == "A" * 50
    assert get_safe_filename("") == "video"
    assert get_safe_filename("   Spaces   ") == "Spaces"

def test_format_duration():
    assert format_duration(46.0) == "46s"
    assert format_duration(60.0) == "1:00"
    assert format_duration(125.0) == "2:05"
    assert format_duration(0.0) == "0s"

def test_calculate_video_metrics_high_priority():
    mock_analysis = MagicMock()
    mock_analysis.segments = [1, 2]
    mock_analysis.text_overlays = [1]  # 2 points
    mock_analysis.narrative_script_segments = [1]  # 3 points
    mock_analysis.visual_cues = [1]  # 2 points
    mock_analysis.speed_effects = [1]  # 4 points
    mock_analysis.key_focus_points = []  # 0 points
    # Total = 11 -> 'medium' priority

    metrics = calculate_video_metrics(mock_analysis)
    assert metrics['total_segments'] == 2
    assert metrics['has_text_overlays'] is True
    assert metrics['has_narrative'] is True
    assert metrics['has_visual_effects'] is True
    assert metrics['has_speed_effects'] is True
    assert metrics['complexity_score'] == 11
    assert metrics['processing_priority'] == 'medium'

def test_calculate_video_metrics_low_priority():
    mock_analysis = MagicMock()
    mock_analysis.segments = [1]
    mock_analysis.text_overlays = [1] * 10  # 20 points
    mock_analysis.narrative_script_segments = [1] * 5  # 15 points
    mock_analysis.visual_cues = []
    mock_analysis.speed_effects = []
    mock_analysis.key_focus_points = []
    # Total = 35 -> 'low' priority

    metrics = calculate_video_metrics(mock_analysis)
    assert metrics['complexity_score'] == 35
    assert metrics['processing_priority'] == 'low'

def test_calculate_video_metrics_exception():
    mock_analysis = MagicMock()
    # cause an exception by making property access raise
    type(mock_analysis).segments = property(lambda x: x / 0)

    metrics = calculate_video_metrics(mock_analysis)
    assert metrics['total_segments'] == 0
    assert metrics['complexity_score'] == 0

def test_validate_analysis_completeness_valid():
    mock_analysis = MagicMock()
    mock_analysis.suggested_title = "Title"
    mock_analysis.segments = [1]
    mock_analysis.hashtags = ["#tag"]
    mock_analysis.original_audio_is_key = True
    mock_analysis.narrative_script_segments = []
    mock_analysis.mood = "Happy"

    is_complete, missing = validate_analysis_completeness(mock_analysis)
    assert is_complete is True
    assert missing == []

def test_validate_analysis_completeness_missing_fields():
    mock_analysis = MagicMock()
    mock_analysis.suggested_title = ""
    mock_analysis.segments = []
    mock_analysis.hashtags = []
    mock_analysis.original_audio_is_key = False
    mock_analysis.narrative_script_segments = []
    mock_analysis.mood = ""

    is_complete, missing = validate_analysis_completeness(mock_analysis)
    assert is_complete is False
    assert "suggested_title" in missing
    assert "video_segments" in missing
    assert "hashtags" in missing
    assert "audio_content" in missing
    assert "mood" in missing

def test_validate_analysis_completeness_exception():
    mock_analysis = MagicMock()
    # Trigger an exception by assigning a type that doesn't have .strip()
    mock_analysis.suggested_title = 123

    is_complete, missing = validate_analysis_completeness(mock_analysis)
    assert is_complete is False
    assert "validation_error" in missing

def test_select_and_validate_segments_valid():
    mock_analysis = MagicMock()

    best_segment = MagicMock()
    best_segment.start_seconds = 10.0
    best_segment.end_seconds = 40.0
    best_segment.reason = "Best part"

    other_segment = MagicMock()
    other_segment.start_seconds = 50.0
    other_segment.end_seconds = 70.0
    other_segment.reason = "Other part"

    mock_analysis.best_segment = best_segment
    mock_analysis.segments = [other_segment]

    config = {'video': {'min_short_duration_seconds': 15, 'max_short_duration_seconds': 60}}

    valid_segments = select_and_validate_segments(mock_analysis, config)
    assert len(valid_segments) == 2
    assert valid_segments[0]['start_seconds'] == 10.0
    assert valid_segments[0]['duration_seconds'] == 30.0
    assert valid_segments[1]['start_seconds'] == 50.0
    assert valid_segments[1]['duration_seconds'] == 20.0

def test_select_and_validate_segments_invalid_timing():
    mock_analysis = MagicMock()

    invalid_seg1 = MagicMock()
    invalid_seg1.start_seconds = -5.0
    invalid_seg1.end_seconds = 10.0
    invalid_seg1.reason = "Negative start"

    invalid_seg2 = MagicMock()
    invalid_seg2.start_seconds = 20.0
    invalid_seg2.end_seconds = 10.0
    invalid_seg2.reason = "End before start"

    mock_analysis.best_segment = invalid_seg1
    mock_analysis.segments = [invalid_seg2]

    config = {}

    valid_segments = select_and_validate_segments(mock_analysis, config)
    assert len(valid_segments) == 0

def test_select_and_validate_segments_too_short():
    mock_analysis = MagicMock()

    short_seg = MagicMock()
    short_seg.start_seconds = 10.0
    short_seg.end_seconds = 15.0 # 5s duration
    short_seg.reason = "Too short"

    mock_analysis.best_segment = short_seg
    mock_analysis.segments = []

    config = {'video': {'min_short_duration_seconds': 15}}

    valid_segments = select_and_validate_segments(mock_analysis, config)
    assert len(valid_segments) == 0

def test_select_and_validate_segments_too_short_coverage_exception():
    mock_analysis = MagicMock()

    # 5s duration, end_seconds is 6.0, coverage is 5/6 = 0.83 > 0.7 and 5 > 2.0
    short_seg = MagicMock()
    short_seg.start_seconds = 1.0
    short_seg.end_seconds = 6.0
    short_seg.reason = "Short but covers whole video"

    mock_analysis.best_segment = short_seg
    mock_analysis.segments = []

    config = {'video': {'min_short_duration_seconds': 15}}

    valid_segments = select_and_validate_segments(mock_analysis, config)
    assert len(valid_segments) == 1
    assert valid_segments[0]['duration_seconds'] == 5.0

def test_select_and_validate_segments_too_long_truncated():
    mock_analysis = MagicMock()

    long_seg = MagicMock()
    long_seg.start_seconds = 10.0
    long_seg.end_seconds = 90.0 # 80s duration
    long_seg.reason = "Too long"

    mock_analysis.best_segment = long_seg
    mock_analysis.segments = []

    config = {'video': {'max_short_duration_seconds': 60}}

    valid_segments = select_and_validate_segments(mock_analysis, config)
    assert len(valid_segments) == 1
    assert valid_segments[0]['duration_seconds'] == 60.0
    assert valid_segments[0]['end_seconds'] == 70.0
    assert "truncated" in valid_segments[0]['reason']

def test_select_and_validate_segments_duplicate():
    mock_analysis = MagicMock()

    seg1 = MagicMock()
    seg1.start_seconds = 10.0
    seg1.end_seconds = 30.0
    seg1.reason = "Seg 1"

    seg2 = MagicMock()
    seg2.start_seconds = 10.5
    seg2.end_seconds = 30.5
    seg2.reason = "Seg 2 (duplicate)"

    mock_analysis.best_segment = seg1
    mock_analysis.segments = [seg2]

    config = {'video': {'min_short_duration_seconds': 15}}

    valid_segments = select_and_validate_segments(mock_analysis, config)
    assert len(valid_segments) == 1

def test_select_and_validate_segments_exception():
    mock_analysis = MagicMock()
    type(mock_analysis).best_segment = property(lambda x: x / 0)

    config = {}
    valid_segments = select_and_validate_segments(mock_analysis, config)
    assert len(valid_segments) == 0
