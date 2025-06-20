"""
Music Manager for YouTube Video Creation
Integrates Spotify discovery and download functionality with video processing
"""

import logging
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from src.config.settings import get_config
from src.integrations.spotify_client import SpotifyClient, SpotifyTrack, MusicDownloadConfig
from src.integrations.spotify_downloader import SpotifyDownloader
from src.processing.advanced_audio_processor import AdvancedAudioProcessor
from src.models import VideoAnalysis, NarrativeSegment


@dataclass
class MusicSelection:
    """Represents a selected music track for video"""
    track: SpotifyTrack
    file_path: Path
    energy_match: float  # 0-1 how well it matches video energy
    genre_match: float   # 0-1 how well it matches preferred genres
    duration_fit: float  # 0-1 how well duration fits video


class MusicManager:
    """
    Manages music discovery, download, and integration for YouTube videos
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.spotify_client = SpotifyClient()
        self.spotify_downloader = SpotifyDownloader()
        self.audio_processor = AdvancedAudioProcessor()
        
        # Music preferences
        self.default_config = MusicDownloadConfig(
            max_tracks=20,
            min_popularity=60,
            preferred_genres=["pop", "electronic", "hip-hop", "indie", "alternative"],
            exclude_explicit=True,
            max_duration_seconds=180,
            min_duration_seconds=30
        )
    
    async def get_trending_music(self, country: str = "US", limit: int = 50) -> List[SpotifyTrack]:
        """
        Get trending/popular music from Spotify
        
        Args:
            country: Country code for localized results
            limit: Maximum number of tracks to return
            
        Returns:
            List of trending Spotify tracks with audio features
        """
        try:
            self.logger.info(f"Getting trending music for {country}")
            
            # Get popular tracks from multiple sources
            popular_tracks = await self.spotify_client.get_popular_tracks(country, limit)
            chart_tracks = await self.spotify_client.get_top_chart_tracks(country, limit)
            
            # Combine and deduplicate
            all_tracks = popular_tracks + chart_tracks
            unique_tracks = {}
            for track in all_tracks:
                if track.track_id not in unique_tracks:
                    unique_tracks[track.track_id] = track
            
            # Enhance with audio features
            track_list = list(unique_tracks.values())
            enhanced_tracks = await self.spotify_client.enhance_tracks_with_features(track_list)
            
            # Filter based on config
            filtered_tracks = await self.spotify_client.filter_tracks_by_config(
                enhanced_tracks, self.default_config
            )
            
            self.logger.info(f"Retrieved {len(filtered_tracks)} trending tracks")
            return filtered_tracks[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting trending music: {e}")
            return []
    
    async def download_music_for_video(self, 
                                     video_analysis: VideoAnalysis,
                                     max_tracks: int = 5) -> List[MusicSelection]:
        """
        Download appropriate music for a video based on its analysis
        
        Args:
            video_analysis: Analysis of the video to match music to
            max_tracks: Maximum number of tracks to download
            
        Returns:
            List of downloaded music selections
        """
        try:
            self.logger.info("Downloading music for video analysis")
            
            # Get trending tracks
            trending_tracks = await self.get_trending_music(limit=50)
            
            if not trending_tracks:
                self.logger.warning("No trending tracks found")
                return []
            
            # Score tracks based on video content
            scored_tracks = self._score_tracks_for_video(trending_tracks, video_analysis)
            
            # Select best matches
            best_tracks = sorted(scored_tracks, key=lambda x: x[1], reverse=True)[:max_tracks]
            
            # Download selected tracks
            music_selections = []
            for track, score in best_tracks:
                try:
                    # For this implementation, we'll use the Spotify downloader
                    # In practice, you might want to create individual track downloads
                    self.logger.info(f"Selected track: {track.artist} - {track.name} (Score: {score:.2f})")
                    
                    # Create a temporary playlist with just this track for download
                    # Note: This is a simplified approach - in practice you'd want 
                    # a more sophisticated download system
                    
                    # For now, we'll mark it as selected but not actually download
                    # since spotify_dl works with playlists
                    music_selection = MusicSelection(
                        track=track,
                        file_path=Path(f"music/{track.artist}_{track.name}.mp3"),
                        energy_match=0.8,  # Placeholder values
                        genre_match=0.7,
                        duration_fit=0.9
                    )
                    music_selections.append(music_selection)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process track {track.name}: {e}")
                    continue
            
            self.logger.info(f"Selected {len(music_selections)} music tracks for video")
            return music_selections
            
        except Exception as e:
            self.logger.error(f"Error downloading music for video: {e}")
            return []
    
    def _score_tracks_for_video(self, 
                               tracks: List[SpotifyTrack], 
                               video_analysis: VideoAnalysis) -> List[tuple]:
        """
        Score tracks based on how well they match the video content
        
        Args:
            tracks: List of available tracks
            video_analysis: Video analysis to match against
            
        Returns:
            List of (track, score) tuples
        """
        scored_tracks = []
        
        try:
            # Determine video characteristics
            video_energy = self._calculate_video_energy(video_analysis)
            video_duration = self._estimate_video_duration(video_analysis)
            video_mood = video_analysis.mood.lower() if video_analysis.mood else "neutral"
            
            for track in tracks:
                score = 0.0
                
                # Energy matching (if available)
                if track.energy is not None:
                    energy_diff = abs(track.energy - video_energy)
                    energy_score = 1.0 - energy_diff
                    score += energy_score * 0.3
                
                # Duration matching
                if video_duration > 0:
                    duration_ratio = min(track.duration_seconds, video_duration) / max(track.duration_seconds, video_duration)
                    score += duration_ratio * 0.2
                
                # Popularity boost
                popularity_score = track.popularity / 100.0
                score += popularity_score * 0.2
                
                # Mood matching (basic keyword matching)
                mood_score = self._calculate_mood_match(track, video_mood)
                score += mood_score * 0.2
                
                # Avoid very short or very long tracks
                if 30 <= track.duration_seconds <= 180:
                    score += 0.1
                
                scored_tracks.append((track, score))
            
        except Exception as e:
            self.logger.error(f"Error scoring tracks: {e}")
        
        return scored_tracks
    
    def _calculate_video_energy(self, video_analysis: VideoAnalysis) -> float:
        """Calculate estimated energy level of video (0.0 to 1.0)"""
        try:
            # Base energy from speed effects
            energy = 0.5  # Default medium energy
            
            if hasattr(video_analysis, 'speed_effects') and video_analysis.speed_effects:
                # Higher energy if there are speed effects
                energy += 0.2
            
            if hasattr(video_analysis, 'visual_cues') and video_analysis.visual_cues:
                # More visual effects = higher energy
                energy += len(video_analysis.visual_cues) * 0.05
            
            # Mood influence
            if video_analysis.mood:
                mood = video_analysis.mood.lower()
                if any(word in mood for word in ["exciting", "energetic", "fast", "intense"]):
                    energy += 0.2
                elif any(word in mood for word in ["calm", "peaceful", "slow", "relaxed"]):
                    energy -= 0.2
            
            return max(0.0, min(1.0, energy))
            
        except Exception:
            return 0.5  # Default medium energy
    
    def _estimate_video_duration(self, video_analysis: VideoAnalysis) -> float:
        """Estimate video duration from analysis"""
        try:
            if hasattr(video_analysis, 'best_segment') and video_analysis.best_segment:
                return video_analysis.best_segment.end_seconds - video_analysis.best_segment.start_seconds
            elif hasattr(video_analysis, 'segments') and video_analysis.segments:
                # Use first segment duration
                segment = video_analysis.segments[0]
                return segment.end_seconds - segment.start_seconds
            else:
                return 60.0  # Default 1 minute
        except Exception:
            return 60.0
    
    def _calculate_mood_match(self, track: SpotifyTrack, video_mood: str) -> float:
        """Calculate how well track mood matches video mood"""
        try:
            # Simple keyword-based mood matching
            track_text = f"{track.name} {track.artist}".lower()
            
            mood_keywords = {
                "exciting": ["energy", "power", "electric", "fire", "wild"],
                "calm": ["chill", "peace", "soft", "quiet", "gentle"],
                "happy": ["joy", "celebration", "party", "fun", "bright"],
                "sad": ["tears", "blue", "rain", "goodbye", "broken"],
                "intense": ["hard", "strong", "force", "impact", "heavy"]
            }
            
            score = 0.0
            for mood, keywords in mood_keywords.items():
                if mood in video_mood:
                    for keyword in keywords:
                        if keyword in track_text:
                            score += 0.2
            
            # Valence matching (if available)
            if track.valence is not None:
                if "happy" in video_mood and track.valence > 0.6:
                    score += 0.3
                elif "sad" in video_mood and track.valence < 0.4:
                    score += 0.3
            
            return min(1.0, score)
            
        except Exception:
            return 0.5
    
    async def download_playlist(self, playlist_url: str) -> List[Path]:
        """
        Download a complete Spotify playlist
        
        Args:
            playlist_url: Spotify playlist URL
            
        Returns:
            List of downloaded file paths
        """
        try:
            self.logger.info(f"Downloading playlist: {playlist_url}")
            downloaded_files = self.spotify_downloader.download_playlist(playlist_url)
            self.logger.info(f"Downloaded {len(downloaded_files)} tracks from playlist")
            return downloaded_files
            
        except Exception as e:
            self.logger.error(f"Error downloading playlist: {e}")
            return []
    
    async def download_top_charts(self, country: str = "US", limit: int = 10) -> List[Path]:
        """
        Download top chart music
        
        Args:
            country: Country code
            limit: Maximum tracks to download
            
        Returns:
            List of downloaded file paths
        """
        try:
            import asyncio
            self.logger.info(f"Downloading top {limit} charts for {country}")
            downloaded_files = await asyncio.to_thread(self.spotify_downloader.download_top_charts, country, limit)
            self.logger.info(f"Downloaded {len(downloaded_files)} chart tracks")
            return downloaded_files
            
        except Exception as e:
            self.logger.error(f"Error downloading top charts: {e}")
            return []
    
    def get_available_music(self) -> List[Path]:
        """Get list of all available music files"""
        try:
            music_dir = self.config.paths.music_folder
            music_files = []
            
            for ext in ["*.mp3", "*.wav", "*.m4a", "*.ogg"]:
                music_files.extend(music_dir.glob(ext))
            
            return music_files
            
        except Exception as e:
            self.logger.error(f"Error getting available music: {e}")
            return []