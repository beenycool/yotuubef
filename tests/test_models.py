import pytest
from pydantic import ValidationError
from src.models import SpeedEffect, SoundEffect


def test_speed_effect_valid_time_range():
    # Should not raise any error
    effect = SpeedEffect(start_seconds=1.0, end_seconds=5.0, speed_factor=1.5)
    assert effect.start_seconds == 1.0
    assert effect.end_seconds == 5.0
    assert effect.speed_factor == 1.5


def test_speed_effect_invalid_time_range_end_less_than_start():
    # End time less than start time should fail
    with pytest.raises(ValidationError) as excinfo:
        SpeedEffect(start_seconds=5.0, end_seconds=1.0, speed_factor=1.5)
    assert "End time must be greater than start time" in str(excinfo.value)


def test_speed_effect_invalid_time_range_end_equal_to_start():
    # End time equal to start time should fail
    with pytest.raises(ValidationError) as excinfo:
        SpeedEffect(start_seconds=5.0, end_seconds=5.0, speed_factor=1.5)
    assert "End time must be greater than start time" in str(excinfo.value)


def test_speed_effect_invalid_start_time_does_not_break_validator():
    # If start_seconds is invalid (e.g. negative), pydantic catches it
    # before or during the start_seconds validation.
    # The validate_time_range shouldn't fail with a missing key error.
    with pytest.raises(ValidationError) as excinfo:
        SpeedEffect(start_seconds=-1.0, end_seconds=5.0, speed_factor=1.5)
    assert "start_seconds" in str(excinfo.value)


def test_sound_effect_valid():
    effect = SoundEffect(timestamp_seconds=2.0, effect_name="whoosh")
    assert effect.timestamp_seconds == 2.0
    assert effect.effect_name == "whoosh"
    assert effect.volume == 0.7  # default


def test_sound_effect_invalid_timestamp():
    with pytest.raises(ValidationError) as excinfo:
        SoundEffect(timestamp_seconds=-1.0, effect_name="whoosh")
    assert "timestamp_seconds" in str(excinfo.value)
