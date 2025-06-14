#!/usr/bin/env python3
"""
Unit tests for YouTube Analytics API metrics fix
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from googleapiclient.errors import HttpError

from src.integrations.youtube_client import YouTubeClient


class TestYouTubeAnalyticsMetricsFix:
    """Test cases for YouTube Analytics API metrics fix"""
    
    @pytest.fixture
    def mock_http_error_400_invalid_metrics(self):
        """Create a mock 400 HttpError for invalid metrics"""
        mock_resp = Mock()
        mock_resp.status = 400
        
        error = HttpError(
            resp=mock_resp,
            content=b'{"error": {"code": 400, "message": "Unknown identifier (impressions) given in field parameters.metrics.", "errors": [{"reason": "invalid"}]}}'
        )
        return error
    
    @pytest.fixture
    def youtube_client(self):
        """Create a YouTube client for testing"""
        with patch('src.integrations.youtube_client.GOOGLE_API_AVAILABLE', True):
            client = YouTubeClient()
            return client
    
    @pytest.mark.asyncio
    async def test_new_metrics_request_format(self, youtube_client):
        """Test that the new metrics are requested correctly"""
        
        # Mock the services
        youtube_client.analytics_service = Mock()
        youtube_client._services_initialized = True
        
        # Mock the reports query method
        mock_reports = Mock()
        mock_query = Mock()
        youtube_client.analytics_service.reports.return_value = mock_reports
        mock_reports.query.return_value = mock_query
        
        # Mock successful response
        with patch.object(youtube_client, '_execute_request_async', return_value={'rows': [[1000, 120, 75.3, 5]]}):
            await youtube_client._get_detailed_analytics("test_video_id")
            
            # Verify the correct metrics are requested
            mock_reports.query.assert_called_once()
            call_args = mock_reports.query.call_args
            
            # Check that the new metrics are used instead of the problematic ones
            metrics_param = call_args[1]['metrics']
            assert 'views' in metrics_param
            assert 'estimatedMinutesWatched' in metrics_param
            assert 'averageViewDuration' in metrics_param
            assert 'subscribersGained' in metrics_param
            
            # Ensure problematic metrics are not used
            assert 'impressions' not in metrics_param
            assert 'clicks' not in metrics_param
            assert 'averageViewPercentage' not in metrics_param
    
    @pytest.mark.asyncio
    async def test_response_parsing_with_new_metrics(self, youtube_client):
        """Test that responses with new metrics are parsed correctly"""
        
        youtube_client.analytics_service = Mock()
        youtube_client._services_initialized = True
        
        # Mock response with new metrics data
        mock_response = {
            'rows': [[1000, 150, 85.7, 12]]  # views, estimatedMinutesWatched, averageViewDuration, subscribersGained
        }
        
        with patch.object(youtube_client, '_execute_request_async', return_value=mock_response):
            result = await youtube_client._get_detailed_analytics("test_video_id")
            
            assert result is not None
            assert result['estimated_minutes_watched'] == 150
            assert result['average_view_duration'] == 85.7
            assert result['subscribers_gained'] == 12
            
            # Legacy fields should be present for backward compatibility
            assert 'impressions' in result
            assert 'clicks' in result
            assert 'ctr' in result
            assert 'average_view_percentage' in result
            
            # Legacy fields should have default values
            assert result['impressions'] == 0
            assert result['clicks'] == 0
            assert result['ctr'] == 0.0
            assert result['average_view_percentage'] == 0.0
    
    @pytest.mark.asyncio
    async def test_backward_compatibility_in_get_video_analytics(self, youtube_client):
        """Test that get_video_analytics maintains backward compatibility"""
        
        # Mock basic video info
        mock_video_info = {
            'statistics': {
                'viewCount': '2000',
                'likeCount': '100',
                'commentCount': '25'
            }
        }
        
        # Mock detailed analytics with new metrics
        mock_detailed_analytics = {
            'estimated_minutes_watched': 200,
            'average_view_duration': 90.5,
            'subscribers_gained': 8,
            'impressions': 0,
            'clicks': 0,
            'ctr': 0.0,
            'average_view_percentage': 0.0
        }
        
        with patch.object(youtube_client, 'get_video_info', return_value=mock_video_info):
            with patch.object(youtube_client, '_get_detailed_analytics', return_value=mock_detailed_analytics):
                
                result = await youtube_client.get_video_analytics("test_video_id")
                
                # Basic stats should be present
                assert result['views'] == 2000
                assert result['likes'] == 100
                assert result['comments'] == 25
                
                # New metrics should be present
                assert result['estimated_minutes_watched'] == 200
                assert result['average_view_duration'] == 90.5
                assert result['subscribers_gained'] == 8
                
                # Legacy fields should still be present
                assert 'impressions' in result
                assert 'clicks' in result
                assert 'ctr' in result
                assert 'average_view_percentage' in result
    
    @pytest.mark.asyncio
    async def test_empty_response_handling(self, youtube_client):
        """Test handling of empty analytics responses"""
        
        youtube_client.analytics_service = Mock()
        youtube_client._services_initialized = True
        
        # Mock empty response
        mock_response = {'rows': []}
        
        with patch.object(youtube_client, '_execute_request_async', return_value=mock_response):
            result = await youtube_client._get_detailed_analytics("test_video_id")
            
            # Should return None for empty response
            assert result is None
    
    @pytest.mark.asyncio
    async def test_incomplete_data_handling(self, youtube_client):
        """Test handling of incomplete data rows"""
        
        youtube_client.analytics_service = Mock()
        youtube_client._services_initialized = True
        
        # Mock response with incomplete data (only 2 values instead of 4)
        mock_response = {
            'rows': [[1000, 120]]  # Missing averageViewDuration and subscribersGained
        }
        
        with patch.object(youtube_client, '_execute_request_async', return_value=mock_response):
            result = await youtube_client._get_detailed_analytics("test_video_id")
            
            assert result is not None
            assert result['estimated_minutes_watched'] == 120
            # Missing values should default to 0
            assert result['average_view_duration'] == 0.0
            assert result['subscribers_gained'] == 0
    
    @pytest.mark.asyncio
    async def test_no_400_error_with_new_metrics(self, youtube_client):
        """Test that the 400 'Unknown identifier' error no longer occurs"""
        
        youtube_client.analytics_service = Mock()
        youtube_client._services_initialized = True
        
        # This test verifies that we don't get the original error
        # by checking that the request is made with valid metrics
        mock_reports = Mock()
        mock_query = Mock()
        youtube_client.analytics_service.reports.return_value = mock_reports
        mock_reports.query.return_value = mock_query
        
        # Mock successful response to simulate the fix working
        with patch.object(youtube_client, '_execute_request_async', return_value={'rows': [[1000, 120, 75.3, 5]]}):
            result = await youtube_client._get_detailed_analytics("test_video_id")
            
            # Should succeed without error
            assert result is not None
            
            # Verify that the metrics parameter doesn't contain problematic identifiers
            call_args = mock_reports.query.call_args
            metrics_param = call_args[1]['metrics']
            
            # These were the problematic metrics that caused the 400 error
            assert 'impressions' not in metrics_param
            assert 'clicks' not in metrics_param
            assert 'averageViewPercentage' not in metrics_param


if __name__ == "__main__":
    pytest.main([__file__, "-v"])