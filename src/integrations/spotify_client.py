"""
Spotify Music Discovery and Download Client for YouTube Video Creation
Handles Spotify API integration, music discovery, and legal download management.
"""

import logging
import asyncio
import aiohttp
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlencode, quote
import base64
import hashlib

from src.config.settings import get_config
from src.models import SpotifyTrack, SpotifyPlaylist, MusicDownloadConfig


# Using Pydantic models from src.models instead of dataclasses


class SpotifyClient:
    """
    Spotify API client for music discovery and metadata extraction.
    Provides legal access to Spotify's catalog for music discovery purposes.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Spotify API configuration with graceful fallback
        try:
            config_client_id = getattr(self.config, 'spotify', {}).get('client_id')
            config_client_secret = getattr(self.config, 'spotify', {}).get('client_secret')
            env_client_id = os.getenv('SPOTIFY_CLIENT_ID')
            env_client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')

            self.client_id = config_client_id
            if env_client_id is not None and env_client_id.strip() != "":
                self.client_id = env_client_id.strip()

            self.client_secret = config_client_secret
            if env_client_secret is not None and env_client_secret.strip() != "":
                self.client_secret = env_client_secret.strip()
        except (AttributeError, TypeError):
            # Fallback to environment variables if config doesn't have spotify section
            env_client_id = os.getenv('SPOTIFY_CLIENT_ID')
            env_client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
            self.client_id = env_client_id.strip() if env_client_id is not None and env_client_id.strip() != "" else None
            self.client_secret = env_client_secret.strip() if env_client_secret is not None and env_client_secret.strip() != "" else None

        if not self.client_id or not self.client_secret:
            self.logger.warning("Spotify API credentials not found. Spotify features will be disabled.")
            self.logger.info("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables to enable Spotify features")
        self.base_url = "https://api.spotify.com/v1"
        self.auth_url = "https://accounts.spotify.com/api/token"
        
        # Authentication state
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        
        # Rate limiting
        self.request_count = 0
        self.rate_limit_window = 60  # seconds
        self.max_requests_per_window = 100
        self.last_request_time = datetime.now()
        
        # Caching
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 3600  # 1 hour
        
    async def authenticate(self) -> bool:
        """Authenticate with Spotify API using Client Credentials flow"""
        try:
            if self.access_token and self.token_expires_at and datetime.now() < self.token_expires_at:
                return True
            
            # Prepare authentication
            auth_string = f"{self.client_id}:{self.client_secret}"
            auth_bytes = auth_string.encode("ascii")
            auth_b64 = base64.b64encode(auth_bytes).decode("ascii")
            
            headers = {
                "Authorization": f"Basic {auth_b64}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            data = {"grant_type": "client_credentials"}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.auth_url, headers=headers, data=data) as response:
                    if response.status == 200:
                        auth_data = await response.json()
                        self.access_token = auth_data["access_token"]
                        expires_in = auth_data.get("expires_in", 3600)
                        self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)
                        
                        self.logger.info("Successfully authenticated with Spotify API")
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Spotify authentication failed: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Error authenticating with Spotify: {e}")
            return False
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated request to Spotify API with rate limiting"""
        try:
            # Check rate limiting
            if not await self._check_rate_limit():
                self.logger.warning("Rate limit exceeded, waiting...")
                await asyncio.sleep(1)
                return None
            
            # Ensure authentication
            if not await self.authenticate():
                return None
            
            # Check cache
            cache_key = f"{endpoint}_{params}"
            if cache_key in self.cache:
                cached_data, cached_time = self.cache[cache_key]
                if datetime.now() - cached_time < timedelta(seconds=self.cache_ttl):
                    return cached_data
            
            # Prepare request
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            # Network timeout and retry configuration
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                try:
                    async with session.get(url, headers=headers, params=params) as response:
                        self.request_count += 1
                        
                        if response.status == 200:
                            data = await response.json()
                            # Cache successful response
                            self.cache[cache_key] = (data, datetime.now())
                            return data
                        elif response.status == 429:
                            # Rate limited
                            retry_after = int(response.headers.get("Retry-After", 1))
                            self.logger.warning(f"Rate limited, waiting {retry_after} seconds")
                            await asyncio.sleep(retry_after)
                            return await self._make_request(endpoint, params)
                        elif response.status in [502, 503, 504]:
                            # Server errors - retry with backoff
                            self.logger.warning(f"Server error {response.status}, will retry")
                            await asyncio.sleep(2)
                            return None  # Let caller handle retry
                        elif response.status == 401:
                            # Unauthorized - clear token and retry once
                            self.logger.warning("Unauthorized - clearing token and retrying")
                            self.access_token = None
                            self.token_expires_at = None
                            return None
                        else:
                            error_text = await response.text()
                            self.logger.error(f"Spotify API error: {response.status} - {error_text}")
                            return None
                            
                except aiohttp.ClientError as e:
                    self.logger.warning(f"Network error in Spotify API request: {e}")
                    return None
                        
        except asyncio.TimeoutError:
            self.logger.warning("Spotify API request timed out")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error making Spotify API request: {e}")
            return None
    
    async def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.now()
        if (now - self.last_request_time).total_seconds() > self.rate_limit_window:
            self.request_count = 0
            self.last_request_time = now
        
        return self.request_count < self.max_requests_per_window
    
    async def get_popular_tracks(self, 
                               country: str = "US", 
                               limit: int = 50,
                               time_range: str = "short_term") -> List[SpotifyTrack]:
        """
        Get popular tracks from Spotify's featured playlists and charts
        
        Args:
            country: Country code for localized results
            limit: Maximum number of tracks to return
            time_range: Time range for popularity (short_term, medium_term, long_term)
            
        Returns:
            List of popular Spotify tracks
        """
        try:
            popular_tracks = []
            
            # Get featured playlists
            featured_data = await self._make_request(
                "browse/featured-playlists",
                {"country": country, "limit": 20}
            )
            
            if featured_data and "playlists" in featured_data:
                for playlist in featured_data["playlists"]["items"][:5]:  # Top 5 playlists
                    playlist_tracks = await self.get_playlist_tracks(playlist["id"])
                    popular_tracks.extend(playlist_tracks[:10])  # Top 10 from each
            
            # Get new releases
            new_releases_data = await self._make_request(
                "browse/new-releases",
                {"country": country, "limit": 20}
            )
            
            if new_releases_data and "albums" in new_releases_data:
                for album in new_releases_data["albums"]["items"][:3]:  # Top 3 albums
                    album_tracks = await self.get_album_tracks(album["id"])
                    popular_tracks.extend(album_tracks[:5])  # Top 5 from each
            
            # Remove duplicates and sort by popularity
            unique_tracks = {}
            for track in popular_tracks:
                if track.track_id not in unique_tracks:
                    unique_tracks[track.track_id] = track
            
            sorted_tracks = sorted(
                unique_tracks.values(),
                key=lambda x: x.popularity,
                reverse=True
            )
            
            self.logger.info(f"Retrieved {len(sorted_tracks)} popular tracks")
            return sorted_tracks[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting popular tracks: {e}")
            return []
    
    async def get_playlist_tracks(self, playlist_id: str) -> List[SpotifyTrack]:
        """Get tracks from a specific playlist"""
        try:
            tracks_data = await self._make_request(
                f"playlists/{playlist_id}/tracks",
                {"market": "US", "limit": 50}
            )
            
            if not tracks_data or "items" not in tracks_data:
                return []
            
            tracks = []
            for item in tracks_data["items"]:
                if item["track"] and item["track"]["type"] == "track":
                    track = await self._parse_track_data(item["track"])
                    if track:
                        tracks.append(track)
            
            return tracks
            
        except Exception as e:
            self.logger.error(f"Error getting playlist tracks: {e}")
            return []
    
    async def get_album_tracks(self, album_id: str) -> List[SpotifyTrack]:
        """Get tracks from a specific album"""
        try:
            tracks_data = await self._make_request(
                f"albums/{album_id}/tracks",
                {"market": "US", "limit": 50}
            )
            
            if not tracks_data or "items" not in tracks_data:
                return []
            
            # Get album info for additional metadata
            album_data = await self._make_request(f"albums/{album_id}")
            album_genres = album_data.get("genres", []) if album_data else []
            
            tracks = []
            for track_data in tracks_data["items"]:
                track = await self._parse_track_data(track_data, album_genres)
                if track:
                    tracks.append(track)
            
            return tracks
            
        except Exception as e:
            self.logger.error(f"Error getting album tracks: {e}")
            return []
    
    async def search_tracks(self, 
                          query: str, 
                          track_type: str = "track",
                          limit: int = 20,
                          gen_z_mode: bool = False) -> List[SpotifyTrack]:
        """Search for tracks by query with Gen Z trending support"""
        try:
            # Apply Gen Z mode if enabled
            if gen_z_mode:
                # Add trending and viral keywords to boost Gen Z appeal
                trending_keywords = ["trending", "viral", "tiktok", "gen z", "popular"]
                query_with_trends = f"{query} {' '.join(trending_keywords)}"
                self.logger.info(f"🎵 Gen Z mode enabled - searching with trending keywords: {query_with_trends}")
            else:
                query_with_trends = query
            
            search_data = await self._make_request(
                "search",
                {
                    "q": query_with_trends,
                    "type": track_type,
                    "market": "US",
                    "limit": limit
                }
            )
            
            if not search_data or "tracks" not in search_data:
                return []
            
            tracks = []
            for track_data in search_data["tracks"]["items"]:
                track = await self._parse_track_data(track_data)
                if track:
                    tracks.append(track)
            
            # Filter for high popularity in Gen Z mode
            if gen_z_mode:
                tracks = [track for track in tracks if track.popularity > 70]
                self.logger.info(f"🎯 Gen Z mode: Filtered to {len(tracks)} high-popularity tracks")
            
            return tracks
            
        except Exception as e:
            self.logger.error(f"Error searching tracks: {e}")
            return []
    
    async def get_track_audio_features(self, track_ids: List[str]) -> Dict[str, Dict]:
        """Get audio features for multiple tracks"""
        try:
            # Spotify allows up to 100 track IDs per request
            batch_size = 100
            all_features = {}
            
            for i in range(0, len(track_ids), batch_size):
                batch_ids = track_ids[i:i + batch_size]
                ids_param = ",".join(batch_ids)
                
                features_data = await self._make_request(
                    "audio-features",
                    {"ids": ids_param}
                )
                
                if features_data and "audio_features" in features_data:
                    for features in features_data["audio_features"]:
                        if features:  # Some tracks might not have features
                            all_features[features["id"]] = features
            
            return all_features
            
        except Exception as e:
            self.logger.error(f"Error getting audio features: {e}")
            return {}
    
    async def _parse_track_data(self, track_data: Dict, album_genres: List[str] = None) -> Optional[SpotifyTrack]:
        """Parse raw track data from Spotify API"""
        try:
            if not track_data or track_data.get("type") != "track":
                return None
            
            # Extract basic info
            track_id = track_data["id"]
            name = track_data["name"]
            duration_ms = track_data.get("duration_ms", 0)
            popularity = track_data.get("popularity", 0)
            preview_url = track_data.get("preview_url")
            external_urls = track_data.get("external_urls", {})
            
            # Extract artist info
            artists = track_data.get("artists", [])
            artist = artists[0]["name"] if artists else "Unknown Artist"
            
            # Extract album info
            album_data = track_data.get("album", {})
            album = album_data.get("name", "Unknown Album")
            genres = album_genres or album_data.get("genres", [])
            
            return SpotifyTrack(
                track_id=track_id,
                name=name,
                artist=artist,
                album=album,
                duration_ms=duration_ms,
                popularity=popularity,
                preview_url=preview_url,
                external_urls=external_urls,
                genres=genres
            )
            
        except Exception as e:
            self.logger.debug(f"Error parsing track data: {e}")
            return None
    
    async def enhance_tracks_with_features(self, tracks: List[SpotifyTrack]) -> List[SpotifyTrack]:
        """Enhance tracks with audio features"""
        try:
            track_ids = [track.track_id for track in tracks]
            features_data = await self.get_track_audio_features(track_ids)
            
            enhanced_tracks = []
            for track in tracks:
                if track.track_id in features_data:
                    features = features_data[track.track_id]
                    track.energy = features.get("energy")
                    track.danceability = features.get("danceability")
                    track.valence = features.get("valence")
                    track.tempo = features.get("tempo")
                
                enhanced_tracks.append(track)
            
            return enhanced_tracks
            
        except Exception as e:
            self.logger.error(f"Error enhancing tracks with features: {e}")
            return tracks
    
    async def filter_tracks_by_config(self, 
                                    tracks: List[SpotifyTrack], 
                                    config: MusicDownloadConfig) -> List[SpotifyTrack]:
        """Filter tracks based on download configuration"""
        try:
            filtered_tracks = []
            
            for track in tracks:
                # Check popularity
                if track.popularity < config.min_popularity:
                    continue
                
                # Check duration
                duration_seconds = track.duration_seconds
                if (duration_seconds < config.min_duration_seconds or 
                    duration_seconds > config.max_duration_seconds):
                    continue
                
                # Check genres if specified
                if config.preferred_genres:
                    track_genres_lower = [g.lower() for g in track.genres]
                    preferred_genres_lower = [g.lower() for g in config.preferred_genres]
                    
                    if not any(genre in track_genres_lower for genre in preferred_genres_lower):
                        # Check if any preferred genre is in track name or artist
                        track_text = f"{track.name} {track.artist}".lower()
                        if not any(genre in track_text for genre in preferred_genres_lower):
                            continue
                
                filtered_tracks.append(track)
                
                # Limit results
                if len(filtered_tracks) >= config.max_tracks:
                    break
            
            self.logger.info(f"Filtered {len(tracks)} tracks to {len(filtered_tracks)} tracks")
            return filtered_tracks
            
        except Exception as e:
            self.logger.error(f"Error filtering tracks: {e}")
            return tracks[:config.max_tracks]
    
    async def get_top_chart_tracks(self,
                                   country: str = "US",
                                   limit: int = 50) -> List[SpotifyTrack]:
        """
        Get top chart tracks from Spotify's featured playlists and categories (official API).
        
        Args:
            country: Country code for localized results
            limit: Maximum number of tracks to return
            
        Returns:
            List of top Spotify tracks
        """
        try:
            # Fetch featured playlists for the country
            featured_data = await self._make_request(
                "browse/featured-playlists",
                {"country": country, "limit": 10}
            )
            playlist_ids = []
            if featured_data and "playlists" in featured_data:
                playlist_ids = [pl["id"] for pl in featured_data["playlists"]["items"]]

            # Optionally, fetch playlists for the "toplists" category
            category_data = await self._make_request(
                "browse/categories/toplists/playlists",
                {"country": country, "limit": 5}
            )
            if category_data and "playlists" in category_data:
                playlist_ids += [pl["id"] for pl in category_data["playlists"]["items"]]

            # Deduplicate playlist IDs
            playlist_ids = list(dict.fromkeys(playlist_ids))

            # Gather tracks from these playlists
            all_tracks = []
            for pid in playlist_ids:
                tracks = await self.get_playlist_tracks(pid)
                all_tracks.extend(tracks[:10])  # Take top 10 from each

            # Deduplicate tracks by track_id
            unique_tracks = {}
            for track in all_tracks:
                if track.track_id not in unique_tracks:
                    unique_tracks[track.track_id] = track

            # Sort by popularity
            sorted_tracks = sorted(
                unique_tracks.values(),
                key=lambda x: x.popularity,
                reverse=True
            )

            return sorted_tracks[:limit]

        except Exception as e:
            self.logger.error(f"Error getting top chart tracks: {e}")
            return []

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_entries": len(self.cache),
            "request_count": self.request_count,
            "rate_limit_window": self.rate_limit_window,
            "max_requests_per_window": self.max_requests_per_window,
            "authenticated": self.access_token is not None,
            "token_expires_at": self.token_expires_at.isoformat() if self.token_expires_at else None
        }