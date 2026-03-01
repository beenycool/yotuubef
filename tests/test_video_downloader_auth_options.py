import logging
from pathlib import Path

from src.processing.video_processor import VideoDownloader


def _build_downloader_stub() -> VideoDownloader:
    downloader = VideoDownloader.__new__(VideoDownloader)
    downloader.logger = logging.getLogger(__name__)
    return downloader


def test_parse_cookies_from_browser_variants():
    assert VideoDownloader._parse_cookies_from_browser("chrome") == ("chrome",)
    assert VideoDownloader._parse_cookies_from_browser("firefox:default") == (
        "firefox",
        "default",
    )
    assert VideoDownloader._parse_cookies_from_browser("edge,Profile 1") == (
        "edge",
        "Profile 1",
    )


def test_get_yt_dlp_auth_options_uses_cookiefile(tmp_path: Path, monkeypatch):
    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    monkeypatch.setenv("YTDLP_COOKIES_FILE", str(cookie_file))
    monkeypatch.delenv("YTDLP_COOKIES_FROM_BROWSER", raising=False)

    downloader = _build_downloader_stub()
    opts = downloader._get_yt_dlp_auth_options()

    assert opts["cookiefile"] == str(cookie_file)


def test_get_yt_dlp_auth_options_uses_browser(monkeypatch):
    monkeypatch.delenv("YTDLP_COOKIES_FILE", raising=False)
    monkeypatch.setenv("YTDLP_COOKIES_FROM_BROWSER", "firefox:default")

    downloader = _build_downloader_stub()
    opts = downloader._get_yt_dlp_auth_options()

    assert opts["cookiesfrombrowser"] == ("firefox", "default")
