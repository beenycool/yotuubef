#!/usr/bin/env python3
"""
Unit tests for YouTube Analytics API error handling
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from googleapiclient.errors import HttpError

from src.integrations.youtube_client import YouTubeClient


class TestYouTubeAnalyticsErrorHandling:
    """Test cases for YouTube Analytics API error handling"""
    
    @pytest.fixture
    def mock_http_error_403(self):
        """Create a mock 403 HttpError for testing"""
        mock_resp = Mock()
        mock_resp.status = 403
        
        error = HttpError(
            resp=mock_resp,
            content=b'{"error": {"code": 403, "message": "YouTube Analytics API has not been used", "errors": [{"reason": "accessNotConfigured"}]}}'
        )
        return error
    
    @pytest.fixture
    def youtube_client(self):
        """Create a YouTube client for testing"""
        with patch('src.integrations.youtube_client.GOOGLE_API_AVAILABLE', True):
            client = YouTubeClient()
            return client
    
    @pytest.mark.asyncio
    async def test_detailed_analytics_handles_403_error(self, youtube_client, mock_http_error_403):
        """Test that _get_detailed_analytics handles 403 accessNotConfigured error gracefully"""
        
        # Mock the services
        youtube_client.analytics_service = Mock()
        youtube_client._services_initialized = True
        
        # Mock the _execute_request_async to raise HttpError
        with patch.object(youtube_client, '_execute_request_async', side_effect=mock_http_error_403):
            result = await youtube_client._get_detailed_analytics("test_video_id")
            
            # Should return None instead of crashing
            assert result is None
    
    @pytest.mark.asyncio
    async def test_detailed_analytics_logs_helpful_message(self, youtube_client, mock_http_error_403, caplog):
        """Test that helpful error message is logged for 403 errors"""
        
        # Mock the services
        youtube_client.analytics_service = Mock()
        youtube_client._services_initialized = True
        
        # Mock the _execute_request_async to raise HttpError
        with patch.object(youtube_client, '_execute_request_async', side_effect=mock_http_error_403):
            await youtube_client._get_detailed_analytics("test_video_id")
            
            # Check that helpful message is logged
            assert "YouTube Analytics API is not enabled" in caplog.text
            assert "console.developers.google.com" in caplog.text
    
    @pytest.mark.asyncio
    async def test_get_video_analytics_continues_without_detailed_analytics(self, youtube_client):
        """Test that get_video_analytics continues working even when detailed analytics fail"""
        
        # Mock basic video info
        mock_video_info = {
            'statistics': {
                'viewCount': '1000',
                'likeCount': '50',
                'commentCount': '10'
            }
        }
        
        # Mock get_video_info to return basic stats
        with patch.object(youtube_client, 'get_video_info', return_value=mock_video_info):
            # Mock _get_detailed_analytics to return None (simulating API error)
            with patch.object(youtube_client, '_get_detailed_analytics', return_value=None):
                
                result = await youtube_client.get_video_analytics("test_video_id")
                
                # Should still return basic analytics
                assert result is not None
                assert result['views'] == 1000
                assert result['likes'] == 50
                assert result['comments'] == 10
                # Detailed analytics should have default values
                assert result['impressions'] == 0
                assert result['clicks'] == 0
                assert result['ctr'] == 0.0
    
    @pytest.mark.asyncio
    async def test_analytics_service_not_available(self, youtube_client):
        """Test behavior when analytics service is not available"""
        
        # Set analytics service to None
        youtube_client.analytics_service = None
        youtube_client._services_initialized = True
        
        result = await youtube_client._get_detailed_analytics("test_video_id")
        
        # Should return None gracefully
        assert result is None
    
    def test_execute_request_async_reraises_http_error(self, youtube_client, mock_http_error_403):
        """Test that _execute_request_async re-raises HttpError for specific handling"""
        
        mock_request = Mock()
        mock_request.execute.side_effect = mock_http_error_403
        
        # Should re-raise HttpError
        with pytest.raises(HttpError):
            asyncio.run(youtube_client._execute_request_async(mock_request))
    
    @pytest.mark.asyncio
    async def test_successful_detailed_analytics(self, youtube_client):
        """Test successful detailed analytics retrieval"""
        
        # Mock successful response with new metrics: views, estimatedMinutesWatched, averageViewDuration, subscribersGained
        mock_response = {
            'rows': [[1000, 120, 75.3, 5]]  # views, estimatedMinutesWatched, averageViewDuration, subscribersGained
        }
        
        youtube_client.analytics_service = Mock()
        youtube_client._services_initialized = True
        
        with patch.object(youtube_client, '_execute_request_async', return_value=mock_response):
            result = await youtube_client._get_detailed_analytics("test_video_id")
            
            assert result is not None
            assert result['estimated_minutes_watched'] == 120
            assert result['average_view_duration'] == 75.3
            assert result['subscribers_gained'] == 5
            # Legacy fields should be 0 for backward compatibility
            assert result['impressions'] == 0
            assert result['clicks'] == 0
            assert result['ctr'] == 0.0
            assert result['average_view_percentage'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])