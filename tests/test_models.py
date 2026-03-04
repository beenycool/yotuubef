import pytest
from pydantic import ValidationError
from src.models import TextOverlay

def test_text_overlay_valid_text():
    # Happy path: valid text
    overlay = TextOverlay(
        text="Hello World",
        timestamp_seconds=1.0,
        duration=5.0
    )
    assert overlay.text == "Hello World"

def test_text_overlay_strips_whitespace():
    # Happy path: whitespace is stripped
    overlay = TextOverlay(
        text="  Hello World  ",
        timestamp_seconds=1.0,
        duration=5.0
    )
    assert overlay.text == "Hello World"

def test_text_overlay_empty_string():
    # Empty string should raise validation error
    with pytest.raises(ValidationError) as exc_info:
        TextOverlay(
            text="",
            timestamp_seconds=1.0,
            duration=5.0
        )
    # Check that either the min_length or the custom validator catches it
    assert "Text content cannot be empty or whitespace only" in str(exc_info.value) or "String should have at least 1 character" in str(exc_info.value)

def test_text_overlay_whitespace_only():
    # Whitespace only string should raise validation error
    with pytest.raises(ValidationError) as exc_info:
        TextOverlay(
            text="   ",
            timestamp_seconds=1.0,
            duration=5.0
        )
    assert "Text content cannot be empty or whitespace only" in str(exc_info.value)

def test_text_overlay_too_long():
    # String longer than 200 chars should raise validation error
    long_text = "a" * 201
    with pytest.raises(ValidationError) as exc_info:
        TextOverlay(
            text=long_text,
            timestamp_seconds=1.0,
            duration=5.0
        )
    assert "String should have at most 200 characters" in str(exc_info.value)
