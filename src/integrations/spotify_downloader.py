"""
Spotify Music Downloader for YouTube Video Creation
Uses spotify_dl to download music from Spotify playlists via YouTube
"""

import logging
import subprocess
import os
from pathlib import Path
from typing import List, Optional
import re
import shutil
import tempfile

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
        Download a Spotify playlist using spotify_dl with improved error handling
        
        Args:
            playlist_url: URL of Spotify playlist
            output_dir: Optional output directory (defaults to music folder)
            
        Returns:
            List of downloaded file paths
        """
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                output_dir = output_dir or self.download_dir
                
                # Check if spotify_dl is available
                if not self._check_spotify_dl_available():
                    self.logger.warning("spotify_dl not found. Please install it with: pip install spotify_dl")
                    self.logger.info("Skipping music download - you can add music files manually to the music/ folder")
                    return []
                
                # Create temp directory for downloads
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Build spotify_dl command using the cached working command
                    if self._spotify_dl_command is None:
                        # Fallback if command wasn't cached
                        self._spotify_dl_command = ["spotify_dl"]
                    
                    cmd = self._spotify_dl_command + [
                        "-l", playlist_url,
                        "-o", str(temp_dir),
                        "-mc", "2",  # Reduced cores to avoid network congestion
                        "--format", "mp3"  # Specify format
                    ]
                    
                    # Run the download command with network retry handling
                    self.logger.info(f"Downloading playlist (attempt {attempt + 1}/{max_retries}): {playlist_url}")
                    
                    try:
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=600,  # 10 minute timeout
                            check=False  # Don't raise on non-zero exit
                        )
                        
                        # Check for network-related errors in output
                        error_output = result.stderr.lower() if result.stderr else ""
                        network_errors = [
                            "network", "timeout", "connection", "unavailable",
                            "http error", "ssl", "certificate", "dns"
                        ]
                        
                        is_network_error = any(error in error_output for error in network_errors)
                        
                        if result.returncode != 0:
                            if is_network_error and attempt < max_retries - 1:
                                self.logger.warning(f"Network error detected, retrying in {retry_delay} seconds...")
                                self.logger.debug(f"Error details: {result.stderr}")
                                import time
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                                continue
                            else:
                                self.logger.error(f"Download failed after {attempt + 1} attempts: {result.stderr}")
                                if "No tracks found" in result.stderr:
                                    self.logger.info("Playlist might be empty or private")
                                return []
                        
                        # Move downloaded files to music directory
                        downloaded_files = []
                        for root, _, files in os.walk(temp_dir):
                            for file in files:
                                if file.endswith((".mp3", ".m4a", ".wav", ".flac")):
                                    src_path = Path(root) / file
                                    # Create safe filename
                                    safe_filename = sanitize_filename(file)
                                    dest_path = output_dir / safe_filename
                                    
                                    # Avoid overwriting existing files
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
                        
                        self.logger.info(f"Successfully downloaded {len(downloaded_files)} tracks from {playlist_url}")
                        return downloaded_files
                        
                    except subprocess.TimeoutExpired:
                        if attempt < max_retries - 1:
                            self.logger.warning(f"Download timeout (attempt {attempt + 1}), retrying...")
                            continue
                        else:
                            self.logger.error("Download timeout after all retries")
                            return []
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Error on attempt {attempt + 1}: {e}, retrying...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    self.logger.error(f"Error downloading playlist after {max_retries} attempts: {e}")
                    return []
        
        self.logger.error("All download attempts failed")
        return []
    
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
            
            # Get country-specific top playlist
            country_top = f"37i9dQZEVXbIPWwFssbupI{country.upper()}"
            
            # Download both playlists
            global_files = self.download_playlist(global_top)[:limit//2]
            country_files = self.download_playlist(country_top)[:limit//2]
            
            return global_files + country_files
            
        except Exception as e:
            self.logger.error(f"Error downloading top charts: {e}")
            return []