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
