"""Simple repository secret scan helpers for CI checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List


DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
}

DEFAULT_EXCLUDED_FILES = {
    "secrets.example",
}


@dataclass(frozen=True)
class SecretFinding:
    path: str
    line_number: int
    pattern_name: str


SECRET_PATTERNS = {
    "assigned_secret": re.compile(
        r"(?i)\b(api[_-]?key|token|secret|password|passwd|client_secret)\b\s*[:=]\s*['\"][^'\"]{8,}['\"]"
    ),
    "private_key_block": re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_pat": re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}


def _is_probably_binary(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(2048)
    except OSError:
        return True

    if not chunk:
        return False
    return b"\x00" in chunk


def iter_text_files(root: Path, excluded_dirs: Iterable[str]) -> Iterable[Path]:
    excluded = set(excluded_dirs)
    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if any(part in excluded for part in path.parts):
            continue

        if path.name in DEFAULT_EXCLUDED_FILES:
            continue

        if _is_probably_binary(path):
            continue

        yield path


def scan_repository_for_secrets(
    root: Path, excluded_dirs: Iterable[str] | None = None
) -> List[SecretFinding]:
    scan_root = Path(root)
    blocked_dirs = set(excluded_dirs or DEFAULT_EXCLUDED_DIRS)
    findings: List[SecretFinding] = []

    for path in iter_text_files(scan_root, blocked_dirs):
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue

        relative_path = str(path.relative_to(scan_root))
        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            for pattern_name, pattern in SECRET_PATTERNS.items():
                if pattern.search(stripped):
                    findings.append(
                        SecretFinding(
                            path=relative_path,
                            line_number=line_number,
                            pattern_name=pattern_name,
                        )
                    )

    return findings
