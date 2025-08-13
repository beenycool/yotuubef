# Social Media Integration Guide

This guide covers the comprehensive social media integration features for TikTok, Instagram, and YouTube, enabling cross-platform video distribution with intelligent scheduling and optimization.

## 🚀 Features

### ✨ Multi-Platform Support
- **YouTube**: Full video upload with metadata, thumbnails, and captions
- **TikTok**: Video uploads with trending hashtags and optimal timing
- **Instagram**: Reels, posts, and stories with platform-specific optimization

### 🎯 Intelligent Scheduling
- **Optimal Timing**: Platform-specific best posting times
- **Cross-Posting**: Intelligent delays between platform uploads
- **Scheduled Uploads**: Future-dated content scheduling

### 📊 Analytics & Monitoring
- **Upload Statistics**: Success rates and performance metrics
- **Platform Analytics**: Individual platform performance tracking
- **History Management**: Comprehensive upload history and cleanup

## 🛠️ Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `secrets.example` to `.env` and fill in your credentials:

```bash
# TikTok API Credentials
TIKTOK_API_KEY="your_tiktok_api_key_here"
TIKTOK_API_SECRET="your_tiktok_api_secret_here"
TIKTOK_ACCESS_TOKEN="your_tiktok_access_token_here"
TIKTOK_REFRESH_TOKEN="your_tiktok_refresh_token_here"
TIKTOK_CLIENT_KEY="your_tiktok_client_key_here"

# Instagram API Credentials
INSTAGRAM_USERNAME="your_instagram_username_here"
INSTAGRAM_PASSWORD="your_instagram_password_here"
INSTAGRAM_SESSION_ID="your_instagram_session_id_here"
INSTAGRAM_ACCESS_TOKEN="your_instagram_access_token_here"
INSTAGRAM_CLIENT_ID="your_instagram_client_id_here"
INSTAGRAM_CLIENT_SECRET="your_instagram_client_secret_here"
```

### 3. Update Configuration

Edit `config.yaml` to enable social media features:

```yaml
feature_flags:
  enable_cross_platform_upload: true
  enable_social_media_analytics: true

social_media:
  enable_cross_posting: true
  cross_post_delay: 300  # 5 minutes between platform posts
  
  tiktok:
    enable_auto_upload: true
    enable_auto_hashtags: true
    enable_trending_topics: true
  
  instagram:
    enable_auto_upload: true
    default_media_type: "reel"
    enable_auto_captions: true
```

## 📱 Platform-Specific Setup

### TikTok Setup

1. **Create TikTok Developer Account**
   - Visit [TikTok for Developers](https://developers.tiktok.com/)
   - Create a new app and get API credentials

2. **Configure Authentication**
   - Set up OAuth 2.0 flow
   - Generate access and refresh tokens

3. **Set Rate Limits**
   ```yaml
   tiktok_rate_limit_requests: 100
   tiktok_rate_limit_window: 3600
   ```

### Instagram Setup

1. **Facebook Developer Account**
   - Visit [Facebook for Developers](https://developers.facebook.com/)
   - Create an Instagram Basic Display app

2. **Authentication Methods**
   - **Username/Password**: Direct login (less secure)
   - **Session ID**: More secure, longer-lasting
   - **Access Token**: Graph API integration

3. **Configure Permissions**
   ```yaml
   instagram_rate_limit_requests: 50
   instagram_rate_limit_window: 3600
   ```

## 🎬 Usage Examples

### Command Line Interface

#### Upload to Multiple Platforms

```bash
# Upload to all platforms
python src/social_media_cli.py upload video.mp4 "My Amazing Video" \
  --platforms youtube tiktok instagram \
  --description "Check out this amazing content!" \
  --tags funny viral trending

# Upload with scheduling
python src/social_media_cli.py upload video.mp4 "My Video" \
  --scheduled "2024-01-15T15:00:00" \
  --platforms tiktok instagram
```

#### Check Status & Analytics

```bash
# View upload statistics
python src/social_media_cli.py status

# Get platform information
python src/social_media_cli.py platform tiktok

# Clean up old history
python src/social_media_cli.py cleanup --days 30
```

### Main Application Integration

```bash
# Social media upload via main app
python main.py social upload video.mp4 "My Video" \
  --platforms youtube tiktok \
  --tags entertainment

# Check social media status
python main.py social status

# Get platform info
python main.py social platform instagram
```

### Programmatic Usage

```python
from src.integrations.social_media_manager import (
    create_social_media_manager,
    CrossPlatformVideoMetadata,
    PlatformType
)

# Initialize manager
manager = create_social_media_manager(config)

# Create metadata
metadata = CrossPlatformVideoMetadata(
    title="My Amazing Video",
    description="Check this out!",
    tags=["funny", "viral", "trending"],
    platforms=[PlatformType.YOUTUBE, PlatformType.TIKTOK, PlatformType.INSTAGRAM],
    cross_post_delay=300
)

# Upload to platforms
results = await manager.upload_to_platforms("video.mp4", metadata)

# Check results
for result in results:
    if result.success:
        print(f"✅ {result.platform.value}: {result.share_url}")
    else:
        print(f"❌ {result.platform.value}: {result.error_message}")
```

## ⏰ Optimal Posting Times

### YouTube
- **Best Days**: Monday - Friday
- **Best Hours**: 12 PM, 3 PM, 7 PM, 9 PM UTC
- **Content Type**: Long-form videos, Shorts

### TikTok
- **Best Days**: Tuesday, Thursday, Friday, Saturday
- **Best Hours**: 9 AM, 12 PM, 7 PM, 9 PM UTC
- **Content Type**: Short-form videos, trending content

### Instagram
- **Best Days**: Monday - Friday
- **Best Hours**: 11 AM, 1 PM, 3 PM, 7 PM UTC
- **Content Type**: Reels, posts, stories

## 📊 Platform Limits & Restrictions

### YouTube
- **File Size**: 128 MB
- **Duration**: Up to 12 hours
- **Daily Limit**: 100 uploads
- **Formats**: MP4, MOV, AVI

### TikTok
- **File Size**: 287 MB
- **Duration**: Up to 10 minutes
- **Daily Limit**: 50 uploads
- **Formats**: MP4, MOV, AVI

### Instagram
- **File Size**: 100 MB
- **Duration**: Up to 90 seconds (Reels)
- **Daily Limit**: 25 uploads
- **Formats**: MP4, MOV

## 🔧 Advanced Configuration

### Cross-Platform Optimization

```yaml
social_media:
  # Intelligent cross-posting
  enable_cross_posting: true
  cross_post_delay: 300
  
  # Platform-specific metadata
  youtube:
    enable_auto_thumbnails: true
    enable_auto_captions: true
  
  tiktok:
    enable_trending_topics: true
    enable_auto_hashtags: true
  
  instagram:
    enable_location_tags: false
    enable_auto_captions: true
```

### Rate Limiting & Throttling

```yaml
api:
  # TikTok rate limits
  tiktok_rate_limit_requests: 100
  tiktok_rate_limit_window: 3600
  
  # Instagram rate limits
  instagram_rate_limit_requests: 50
  instagram_rate_limit_window: 3600
```

### Error Handling & Retry Logic

The system includes comprehensive error handling:

- **API Failures**: Automatic retry with exponential backoff
- **Rate Limiting**: Intelligent throttling and queue management
- **Network Issues**: Connection retry and fallback mechanisms
- **Platform Errors**: Detailed error reporting and recovery

## 📈 Analytics & Monitoring

### Upload Statistics

```python
# Get comprehensive statistics
stats = await manager.get_upload_statistics()

print(f"Total Uploads: {stats['total_uploads']}")
print(f"Success Rate: {stats['success_rate']:.1f}%")

# Platform-specific stats
for platform, platform_stats in stats['platform_stats'].items():
    print(f"{platform.value}: {platform_stats['success_rate']:.1f}%")
```

### Upload History

```python
# Get recent uploads
history = await manager.get_upload_history(limit=50)

# Filter by platform
tiktok_uploads = await manager.get_upload_history(
    platform=PlatformType.TIKTOK, 
    limit=25
)
```

## 🚨 Troubleshooting

### Common Issues

#### TikTok API Errors
- **Invalid API Key**: Verify credentials in `.env`
- **Rate Limit Exceeded**: Check `tiktok_rate_limit_requests` setting
- **Token Expired**: Refresh access token using refresh token

#### Instagram API Errors
- **Login Failed**: Check username/password or session ID
- **Permission Denied**: Verify app permissions in Facebook Developer Console
- **Rate Limited**: Reduce upload frequency or increase delays

#### General Issues
- **File Size Too Large**: Check platform-specific limits
- **Unsupported Format**: Convert to MP4 or MOV
- **Network Errors**: Check internet connection and API endpoints

### Debug Mode

Enable debug logging for troubleshooting:

```yaml
development:
  debug_mode: true
  enable_profiling: true
```

### Health Checks

```bash
# Check platform connectivity
python src/social_media_cli.py status

# Test individual platform
python src/social_media_cli.py platform tiktok
```

## 🔒 Security Best Practices

### API Key Management
- Store credentials in `.env` file (never commit to version control)
- Use environment variables in production
- Rotate API keys regularly
- Implement least-privilege access

### Rate Limiting
- Respect platform rate limits
- Implement intelligent throttling
- Monitor API usage and costs
- Use webhooks for real-time updates

### Data Privacy
- Minimize data collection
- Secure storage of user credentials
- Implement proper logging and audit trails
- Follow GDPR and privacy regulations

## 🚀 Performance Optimization

### Upload Efficiency
- **Parallel Processing**: Upload to multiple platforms simultaneously
- **Batch Operations**: Group multiple videos for efficient processing
- **Caching**: Cache API responses and metadata
- **Compression**: Optimize video files for each platform

### Resource Management
- **Memory Usage**: Efficient handling of large video files
- **Network Optimization**: Connection pooling and keep-alive
- **Error Recovery**: Graceful degradation and fallback mechanisms
- **Monitoring**: Real-time performance tracking

## 📚 Additional Resources

### Documentation
- [TikTok for Developers](https://developers.tiktok.com/)
- [Instagram Basic Display API](https://developers.facebook.com/docs/instagram-basic-display-api/)
- [YouTube Data API](https://developers.google.com/youtube/v3)

### Community & Support
- GitHub Issues: Report bugs and request features
- Discord Community: Get help and share experiences
- Documentation Wiki: Comprehensive guides and tutorials

### Updates & Changelog
- Follow repository for latest updates
- Check release notes for new features
- Subscribe to security advisories

---

**Note**: This integration requires valid API credentials for each platform. Ensure compliance with platform terms of service and API usage policies.