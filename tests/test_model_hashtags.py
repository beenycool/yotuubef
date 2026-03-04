import pytest
from pydantic import ValidationError
from src.models import VideoAnalysis

def test_validate_hashtags_valid_formats():
    """Test valid hashtags are preserved and properly formatted."""
    input_tags = ["#awesome", "video", "#cool_stuff", "   #spaces   "]
    expected = ["#awesome", "#video", "#cool_stuff", "#spaces"]

    result = VideoAnalysis.validate_hashtags(input_tags)
    assert result == expected

def test_validate_hashtags_strips_special_characters():
    """Test special characters are stripped but alphanumeric and underscores are kept."""
    standard_tags = ["#hello!world", "#tag@name", "#with_underscore", "#123_abc", "#a#b#c"]
    expected = ["#helloworld", "#tagname", "#with_underscore", "#123_abc", "#a"]

    result = VideoAnalysis.validate_hashtags(standard_tags)
    assert result == expected

def test_validate_hashtags_handles_multiple_hashes():
    """Test handling of tags with multiple hash symbols."""
    input_tags = ["#single", "##double", "#a#b", "no#hash"]
    expected = ["#single", "#a", "#no"]

    result = VideoAnalysis.validate_hashtags(input_tags)
    assert result == expected

def test_validate_hashtags_ignores_empty_and_whitespace():
    """Test empty strings and whitespace-only strings are ignored."""
    input_tags = ["#valid", "", "   ", "\t", "\n"]
    expected = ["#valid"]

    result = VideoAnalysis.validate_hashtags(input_tags)
    assert result == expected

def test_validate_hashtags_raises_value_error_if_empty():
    """Test ValueError is raised if no valid hashtags remain."""
    invalid_inputs = [
        [],
        [""],
        ["   "],
        ["#!@#"],
        ["##invalid"],
        ["#   "]
    ]

    for invalid_input in invalid_inputs:
        with pytest.raises(ValueError, match="At least one valid hashtag is required"):
            VideoAnalysis.validate_hashtags(invalid_input)

def test_validate_hashtags_pydantic_validation():
    """Test via Pydantic model validation to ensure it integrates correctly."""
    # We create a valid subset of data, only replacing hashtags to test validation
    valid_data = {
        "suggested_title": "Test Title",
        "summary_for_description": "Test Summary",
        "mood": "happy",
        "has_clear_narrative": True,
        "original_audio_is_key": False,
        "hook_text": "Test Hook",
        "hook_variations": ["Hook 1"],
        "visual_hook_moment": {"timestamp_seconds": 1.0, "description": "hook"},
        "audio_hook": {"type": "type", "sound_name": "sound", "timestamp_seconds": 1.0},
        "best_segment": {"start_seconds": 0.0, "end_seconds": 5.0, "reason": "reason"},
        "segments": [{"start_seconds": 0.0, "end_seconds": 5.0, "reason": "reason"}],
        "music_genres": ["pop"],
        "hashtags": ["#valid_tag"],  # We will test overriding this below
        "thumbnail_info": {"timestamp_seconds": 1.0, "reason": "reason"},
        "call_to_action": {"text": "Click here", "type": "subscribe"}
    }

    # Test valid instantiation
    model = VideoAnalysis(**valid_data)
    assert model.hashtags == ["#valid_tag"]

    # Test invalid instantiation (empty list)
    invalid_data = valid_data.copy()
    invalid_data["hashtags"] = ["   ", "#!@#"]

    with pytest.raises(ValidationError) as exc_info:
        VideoAnalysis(**invalid_data)

    assert "At least one valid hashtag is required" in str(exc_info.value)
