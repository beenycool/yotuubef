from pathlib import Path

from src.utils.secret_scanner import scan_repository_for_secrets


def test_secret_scanner_detects_assigned_secrets(tmp_path: Path):
    target = tmp_path / "sample.py"
    target.write_text('API_KEY = "super-secret-value"\n', encoding="utf-8")

    findings = scan_repository_for_secrets(tmp_path)

    assert len(findings) == 1
    assert findings[0].path == "sample.py"
    assert findings[0].line_number == 1
    assert findings[0].pattern_name == "assigned_secret"


def test_secret_scanner_ignores_examples_and_comments(tmp_path: Path):
    example_file = tmp_path / "secrets.example"
    example_file.write_text('API_KEY = "placeholder-only"\n', encoding="utf-8")

    source_file = tmp_path / "notes.txt"
    source_file.write_text('# token = "not-real"\nregular text\n', encoding="utf-8")

    findings = scan_repository_for_secrets(tmp_path)
    assert findings == []


def test_secret_scanner_skips_excluded_directories(tmp_path: Path):
    git_file = tmp_path / ".git" / "config"
    git_file.parent.mkdir(parents=True)
    git_file.write_text('password = "leak"\n', encoding="utf-8")

    findings = scan_repository_for_secrets(tmp_path)
    assert findings == []
