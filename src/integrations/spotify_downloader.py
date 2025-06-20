"""
Spotify Music Downloader for YouTube Video Creation
Uses spotify_dl to download music from Spotify playlists via YouTube
"""

import logging
import subprocess
import os
from pathlib import Path
from typing import List, Optional
import shutil
import tempfile
import time
import random

from src.config.settings import get_config
from src.utils.common_utils import sanitize_filename

class SpotifyDownloader:
    """
    Downloads music from Spotify playlists using spotify_dl and yt-dlp
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.download_dir = self.config.paths.music_folder
        self.download_dir.mkdir(exist_ok=True)
        self._spotify_dl_command = None  # Cache the working command
        
    def _check_spotify_dl_available(self) -> bool:
        """Check if spotify_dl command is available"""
        try:
            # Try different ways to check for spotify_dl
            commands_to_try = [
                ["spotify_dl"],
                ["python", "-m", "spotify_dl"],
                ["python3", "-m", "spotify_dl"]
            ]
            
            for base_cmd in commands_to_try:
                try:
                    test_cmd = base_cmd + ["--version"]
                    result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        self.logger.info(f"Found spotify_dl using command: {' '.join(base_cmd)}")
                        self._spotify_dl_command = base_cmd  # Cache the working command
                        return True
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            
            return False
        except Exception as e:
            self.logger.warning(f"Error checking spotify_dl availability: {e}")
            return False

    def download_playlist(self, playlist_url: str, output_dir: Optional[Path] = None) -> List[Path]:
        """
        Download a Spotify playlist using spotify_dl with improved error handling.
        Refactored for readability and maintainability.
        """
        output_dir = output_dir or self.download_dir

        if not self._check_spotify_dl_available_and_log():
            return []

        def attempt():
            # This function signature matches what _retry_with_backoff expects
            return self._attempt_single_download(
                playlist_url, output_dir, 0, 0, 0
            )

        result = self._retry_with_backoff(
            attempt_fn=attempt,
            max_retries=3,
            initial_delay=2
        )
        return result or []

    def _retry_with_backoff(self, attempt_fn, max_retries=3, initial_delay=2):
        """
        Helper to retry a function with exponential backoff and jitter.
        Returns the first element of the tuple from attempt_fn if successful, else None.
        """
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                downloaded_files, error, should_retry = attempt_fn()
                if downloaded_files:
                    return downloaded_files
                if not should_retry:
                    self._handle_download_failure(error, attempt, max_retries)
                    return None
                self.logger.warning(f"Attempt {attempt + 1} failed: {error}. Retrying in {delay}s...")
            except Exception as e:
                error = str(e)
                if attempt < max_retries - 1:
                    self.logger.warning(f"Exception on attempt {attempt + 1}: {e}, retrying in {delay}s...")
                else:
                    self.logger.error(f"Error downloading playlist after {max_retries} attempts: {e}")
                    return None
            time.sleep(delay + random.uniform(-0.2, 0.2) * delay)
            delay *= 2
        self.logger.error("All download attempts failed")
        return None

    def _check_spotify_dl_available_and_log(self) -> bool:
        if not self._check_spotify_dl_available():
            self.logger.warning("spotify_dl not found. Please install it with: pip install spotify_dl")
            self.logger.info("Skipping music download - you can add music files manually to the music/ folder")
            return False
        return True

    def _attempt_single_download(self, playlist_url, output_dir, attempt, max_retries, retry_delay):
        with tempfile.TemporaryDirectory() as temp_dir:
            cmd = self._get_spotify_dl_command(playlist_url, temp_dir)
            self.logger.info(f"Downloading playlist (attempt {attempt + 1}/{max_retries}): {playlist_url}")
            result, timeout = self._run_spotify_dl_command(cmd)
            if timeout:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Download timeout (attempt {attempt + 1}), retrying...")
                    return None, "timeout", True
                else:
                    return None, "timeout", False
            should_retry = self._should_retry_download(result, attempt, max_retries)
            if result and result.returncode == 0:
                downloaded_files = self._move_downloaded_files(temp_dir, output_dir)
                self.logger.info(f"Successfully downloaded {len(downloaded_files)} tracks from {playlist_url}")
                return downloaded_files, None, False
            else:
                return None, result.stderr if result else "unknown error", should_retry

    def _get_spotify_dl_command(self, playlist_url, temp_dir):
        if self._spotify_dl_command is None:
            self._spotify_dl_command = ["spotify_dl"]
        return self._spotify_dl_command + [
            "-l", playlist_url,
            "-o", str(temp_dir),
            "-mc", "2",
            "--format", "mp3"
        ]

    def _run_spotify_dl_command(self, cmd):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                check=False
            )
            return result, False
        except subprocess.TimeoutExpired:
            return None, True

    def _should_retry_download(self, result, attempt, max_retries):
        if not result or result.returncode == 0:
            return False
        error_output = result.stderr.lower() if result.stderr else ""
        network_errors = [
            "network", "timeout", "connection", "unavailable",
            "http error", "ssl", "certificate", "dns"
        ]
        is_network_error = any(error in error_output for error in network_errors)
        return is_network_error and attempt < max_retries - 1

    def _handle_download_failure(self, error, attempt, max_retries):
        if error == "timeout":
            self.logger.error("Download timeout after all retries")
        else:
            self.logger.error(f"Download failed after {attempt + 1} attempts: {error}")
            if error and "No tracks found" in error:
                self.logger.info("Playlist might be empty or private")

    def _move_downloaded_files(self, temp_dir, output_dir):
        downloaded_files = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith((".mp3", ".m4a", ".wav", ".flac")):
                    src_path = Path(root) / file
                    safe_filename = sanitize_filename(file)
                    dest_path = output_dir / safe_filename
                    counter = 1
                    while dest_path.exists():
                        name, ext = safe_filename.rsplit('.', 1)
                        dest_path = output_dir / f"{name}_{counter}.{ext}"
                        counter += 1
                    try:
                        shutil.move(str(src_path), str(dest_path))
                        downloaded_files.append(dest_path)
                        self.logger.debug(f"Downloaded: {dest_path.name}")
                    except Exception as move_error:
                        self.logger.warning(f"Failed to move file {file}: {move_error}")
        return downloaded_files
    def download_top_charts(self, country: str = "US", limit: int = 10) -> List[Path]:
        """
        Download top charts using Spotify's official playlist IDs
        
        Args:
            country: Country code (2 letters)
            limit: Maximum tracks to download
            
        Returns:
            List of downloaded file paths
        """
        try:
            # Get global top 50 playlist
            global_top = "37i9dQZEVXbMDoHDwVN2tF"
            global_files = self.download_playlist(global_top)[:limit//2]
            
            # Get country-specific top playlist
            try:
                # Correct country-specific playlist format
                country_top = f"37i9dQZEVXbL0GavIqMTeb{country.upper()}"
                country_files = self.download_playlist(country_top)[:limit//2]
            except Exception as country_error:
                self.logger.warning(f"Couldn't download country charts: {country_error}")
                # Fallback to global charts if country-specific fails
                country_files = self.download_playlist(global_top)[:limit//2]
            
            return global_files + country_files
            
        except Exception as e:
            self.logger.error(f"Error downloading top charts: {e}")
            return []