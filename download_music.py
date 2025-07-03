#!/usr/bin/env python3
"""
Simple script to download popular music using the Spotify integration
"""

import asyncio
import logging
from pathlib import Path

from src.processing.music_manager import MusicManager
from src.integrations.spotify_downloader import SpotifyDownloader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Download popular music"""
    print("🎵 Starting Music Download...")
    
    # Initialize downloader (simpler approach without full Spotify API)
    downloader = SpotifyDownloader()
    
    # Download top charts (most streamed)
    print("📈 Downloading Top Charts...")
    top_files = await asyncio.to_thread(downloader.download_top_charts, "US", 10)
    
    if top_files:
        print(f"✅ Downloaded {len(top_files)} top chart tracks:")
        for file_path in top_files:
            print(f"   🎶 {file_path.name}")
    else:
        print("❌ No tracks downloaded from top charts")
    
    # You can also download specific playlists
    print("\n🎵 Want to download a specific playlist?")
    print("Usage examples:")
    print("  - Global Top 50: https://open.spotify.com/playlist/37i9dQZEVXbMDoHDwVN2tF")
    print("  - Today's Top Hits: https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M")
    
    # Example: Download a specific popular playlist
    # popular_playlist = "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"
    # playlist_files = await asyncio.to_thread(downloader.download_playlist, popular_playlist)
    
    print("\n🎉 Music download complete!")
    print("📁 Files saved to: music/")

def download_popular_playlists():
    """Download from popular Spotify playlists"""
    downloader = SpotifyDownloader()
    
    popular_playlists = [
        "37i9dQZEVXbMDoHDwVN2tF",  # Global Top 50
        "37i9dQZF1DXcBWIGoYBM5M",  # Today's Top Hits
        "37i9dQZF1DX0XUsuxWHRQd",  # RapCaviar
        "37i9dQZF1DWXRqgorJj26U",  # Rock Classics
        "37i9dQZF1DX4dyzvuaRJ0n",  # mint
    ]
    
    all_downloads = []
    for playlist_id in popular_playlists:
        try:
            playlist_url = f"https://open.spotify.com/playlist/{playlist_id}"
            files = downloader.download_playlist(playlist_url)
            all_downloads.extend(files)
            print(f"Downloaded {len(files)} tracks from playlist {playlist_id}")
        except Exception as e:
            print(f"Failed to download playlist {playlist_id}: {e}")
    
    return all_downloads

if __name__ == "__main__":
    print("🎵 Spotify Music Downloader")
    print("=" * 50)
    
    # Option 1: Download top charts
    print("1. Download Top Charts (10 tracks)")
    print("2. Download Popular Playlists (50+ tracks)")
    
    choice = input("\nSelect option (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(main())
    elif choice == "2":
        files = download_popular_playlists()
        print(f"\n🎉 Downloaded {len(files)} total tracks!")
    else:
        print("❌ Invalid choice. Run script again.")