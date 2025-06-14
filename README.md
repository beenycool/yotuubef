# Enhanced AI-Powered YouTube Shorts Generator

An advanced, AI-driven system for automatically creating engaging YouTube Shorts from Reddit content, now powered by **Google Gemini API** for superior AI analysis and content optimization.

## ✨ Key Features

### 🎬 **Cinematic Video Processing**
- **Dynamic Camera Movements**: AI-powered zoom, pan, and rotation effects
- **Scene Analysis**: Intelligent detection of visual complexity and motion
- **Composition Optimization**: Rule of thirds, symmetry, and contrast analysis
- **Cinematic Transitions**: Smooth transitions between scenes

### 🎵 **Advanced Audio Processing**
- **Intelligent Audio Ducking**: Smart background music volume adjustment during speech
- **Voice Enhancement**: Noise reduction, clarity boost, and dynamic compression
- **Audio Analysis**: Real-time frequency analysis and rhythm preservation
- **Multi-track Mixing**: Professional-grade audio layering

### 🖼️ **Smart Thumbnail Generation & A/B Testing**
- **Multiple Variants**: Generate 3-5 thumbnail variants automatically
- **A/B Testing**: Performance-driven thumbnail optimization
- **Style Variations**: Different color schemes, text styles, and emotional tones
- **Performance Analytics**: Track click-through rates and engagement

### 🤖 **Google Gemini AI Integration**
- **Advanced Content Analysis**: Powered by Gemini 2.5 Flash Preview
- **Smart Rate Limiting**: Built-in 10 RPM / 500 daily request management
- **Intelligent Optimization**: AI-driven content suggestions and improvements
- **Fallback Support**: Graceful degradation when API limits are reached

### 📊 **Performance Optimization**
- **Auto-Parameter Tuning**: Statistical analysis of video performance
- **Enhancement Tracking**: Monitor the impact of AI improvements
- **Performance Metrics**: Comprehensive analytics and reporting
- **Optimization Cycles**: Automated improvement suggestions

### 💬 **Proactive Channel Management**
- **Comment Analysis**: AI-powered engagement scoring and sentiment analysis
- **Automated Interactions**: Smart comment responses and moderation
- **Engagement Optimization**: Identify and promote high-value interactions
- **Community Building**: Proactive audience engagement strategies

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Google Gemini API Key** (get from [Google AI Studio](https://aistudio.google.com/app/apikey))
- **Reddit API Credentials**
- **YouTube API Credentials**
- **FFmpeg** (for video processing)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd youtube-shorts-generator
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements_enhanced.txt
   ```

3. **Set up credentials:**
   ```bash
   cp secrets.example .env
   # Edit .env with your API keys
   ```

4. **Configure the system:**
   ```bash
   # Edit config_enhanced.yaml for custom settings
   ```

### Configuration

#### Essential API Keys (Required)

Add these to your `.env` file:

```env
# Google Gemini API (ONLY AI service used)
GEMINI_API_KEY="your_gemini_api_key_here"

# Reddit API
REDDIT_CLIENT_ID="your_reddit_client_id"
REDDIT_CLIENT_SECRET="your_reddit_client_secret"
REDDIT_USER_AGENT="python:VideoBot:v2.0 (by /u/yourusername)"

# YouTube API
YOUTUBE_API_KEY="your_youtube_api_key"
GOOGLE_CLIENT_SECRETS_FILE="client_secret_your_project.json"
YOUTUBE_TOKEN_FILE="youtube_token.json"
```

#### Gemini API Rate Limits

The system is configured for:
- **10 requests per minute (RPM)**
- **500 requests per day**
- **Automatic rate limiting** with intelligent queuing

## 📋 Usage

### Basic Video Generation

```bash
# Generate a single video
python main_enhanced.py --url "https://reddit.com/r/oddlysatisfying/comments/xyz"

# Batch process multiple videos
python main_enhanced.py --batch --subreddit "oddlysatisfying" --count 5

# With custom settings
python main_enhanced.py --url "reddit_url" --enable-ab-testing --optimize-performance
```

### Advanced Features

```bash
# Enable all AI features
python main_enhanced.py \
  --url "reddit_url" \
  --enable-cinematic \
  --enable-advanced-audio \
  --enable-ab-testing \
  --enable-optimization \
  --enable-management

# Performance monitoring mode
python main_enhanced.py --monitor-performance --optimization-cycle 7

# Channel management only
python main_enhanced.py --manage-channel --analyze-comments
```

## 🏗️ Architecture

### Core Components

1. **Enhanced Orchestrator** (`src/enhanced_orchestrator.py`)
   - Coordinates all AI-enhanced processing steps
   - Manages workflow between components

2. **Gemini AI Client** (`src/integrations/gemini_ai_client.py`)
   - Google Gemini API integration with rate limiting
   - Intelligent content analysis and optimization

3. **Cinematic Editor** (`src/processing/cinematic_editor.py`)
   - AI-powered video effects and camera movements
   - Scene analysis and composition optimization

4. **Advanced Audio Processor** (`src/processing/advanced_audio_processor.py`)
   - Intelligent audio ducking and enhancement
   - Multi-track audio processing

5. **Enhanced Thumbnail Generator** (`src/processing/enhanced_thumbnail_generator.py`)
   - A/B testing and performance analytics
   - Multiple style variations

6. **Enhancement Optimizer** (`src/processing/enhancement_optimizer.py`)
   - Performance tracking and auto-optimization
   - Statistical analysis and suggestions

7. **Channel Manager** (`src/management/channel_manager.py`)
   - Proactive comment management
   - Audience engagement optimization

### AI Integration Flow

```
Reddit Content → Gemini Analysis → Video Processing
      ↓                ↓               ↓
Performance Data → Optimization → Enhanced Output
      ↓                ↓               ↓
Analytics → AI Feedback → Continuous Improvement
```

## ⚙️ Configuration

### Gemini AI Settings

```yaml
# config_enhanced.yaml
api:
  gemini_api_key: "your_key"
  gemini_model: "gemini-2.5-flash-preview-05-20"
  gemini_rate_limit_rpm: 10
  gemini_rate_limit_daily: 500

ai_features:
  enable_cinematic_editing: true
  enable_advanced_audio: true
  enable_ab_testing: true
  enable_auto_optimization: true
  enable_proactive_management: true
```

### Performance Optimization

```yaml
optimization:
  enable_auto_optimization: true
  confidence_threshold: 0.7
  max_adjustment_per_cycle: 0.2
  optimization_cycle_days: 7
  min_sample_size: 10
```

### A/B Testing

```yaml
thumbnail_optimization:
  enable_ab_testing: true
  ab_test_variants: 3
  test_duration_hours: 24
  min_impressions_for_test: 1000
```

## 📊 Performance & Analytics

### Metrics Tracked

- **Engagement Scores**: AI-calculated engagement potential
- **Retention Predictions**: Intro, mid, and end retention forecasts
- **Click-through Rates**: Thumbnail performance analytics
- **Comment Sentiment**: AI-powered sentiment analysis
- **Performance Trends**: Long-term optimization tracking

### Optimization Strategies

1. **Parameter Tuning**: Automatic adjustment of video effects based on performance
2. **Content Analysis**: AI-driven content optimization suggestions
3. **Thumbnail Testing**: Data-driven thumbnail variant selection
4. **Engagement Optimization**: Proactive audience interaction strategies

## 🛠️ Development

### Project Structure

```
src/
├── config/              # Configuration management
├── integrations/        # API integrations (Gemini, Reddit, YouTube)
├── processing/          # Video/audio processing modules
├── management/          # Channel and performance management
├── database/           # Data storage and analytics
└── models.py           # Data models and types
```

### Adding New Features

1. Create your feature module in the appropriate `src/` subdirectory
2. Update `enhanced_orchestrator.py` to integrate the feature
3. Add configuration options to `config_enhanced.yaml`
4. Update the main entry point in `main_enhanced.py`

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Test with mock data
python main_enhanced.py --test-mode --mock-ai

# Performance testing
python main_enhanced.py --profile --test-video-limit 60
```

## 🔧 Troubleshooting

### Common Issues

1. **Gemini API Rate Limits**
   - The system automatically handles rate limiting
   - Monitor daily usage in logs
   - Consider upgrading API quota if needed

2. **Video Processing Errors**
   - Check FFmpeg installation
   - Verify GPU acceleration settings
   - Monitor memory usage

3. **Authentication Issues**
   - Verify all API keys are correct
   - Check YouTube OAuth token validity
   - Ensure proper file permissions

4. **YouTube Analytics API Error (403 accessNotConfigured)**
   ```
   ERROR - Google API request failed: HttpError 403 when requesting
   https://youtubeanalytics.googleapis.com/v2/reports?...
   "YouTube Analytics API has not been used in project before or it is disabled"
   ```
   
   **Solution:**
   ```bash
   # Run the diagnostic script
   python check_youtube_analytics_api.py
   
   # Follow the steps to:
   # 1. Enable YouTube Analytics API in Google Cloud Console
   # 2. Re-authenticate with updated scopes
   # 3. Wait for API activation
   ```
   
   **Manual Steps:**
   - Go to [Google Cloud Console](https://console.developers.google.com/apis/api/youtubeanalytics.googleapis.com/overview)
   - Select your project and click "Enable"
   - Delete `youtube_token.json` and run `python auth_youtube.py`
   - Wait 2-5 minutes for API activation

### Logging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main_enhanced.py --debug

# View comprehensive logs
tail -f logs/enhanced_generator.log
```

## 📈 Performance Optimization

### GPU Acceleration

```yaml
performance:
  enable_gpu_acceleration: true
  max_vram_usage: 0.7
  gpu_memory_management: true
  clear_cache_between_videos: true
```

### Memory Management

```yaml
performance:
  max_concurrent_videos: 3
  chunk_size_mb: 100
  aggressive_memory_cleanup: true
  memory_monitoring_interval: 30
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆕 Migration from OpenAI

This version has been fully migrated from OpenAI to Google Gemini API for enhanced performance and cost efficiency:

### Key Changes
- **Replaced OpenAI GPT** with **Gemini 2.5 Flash Preview**
- **Built-in rate limiting** (10 RPM / 500 daily)
- **Enhanced content analysis** with improved accuracy
- **Cost-effective processing** with generous rate limits
- **Fallback support** for graceful degradation

### Migration Benefits
- **Lower costs** compared to OpenAI
- **Higher rate limits** for better throughput
- **Improved AI analysis** quality
- **Better integration** with Google services

---

**Ready to create amazing YouTube Shorts? Get started with the enhanced AI-powered generator today!** 🚀