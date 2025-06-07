# YouTube Shorts Video Generation System v2.0

An automated system for generating engaging YouTube Shorts from Reddit content using AI-powered analysis and video processing.

## 🚀 Features

- **Automated Content Discovery**: Fetches video content from curated Reddit subreddits
- **AI-Powered Analysis**: Uses Google Gemini to analyze videos and generate enhancement suggestions
- **Smart Video Processing**: Applies effects, text overlays, and audio enhancements automatically
- **YouTube Integration**: Automatically uploads processed videos with optimized metadata
- **Content Filtering**: Built-in safety checks for family-friendly, monetizable content
- **Database Tracking**: Comprehensive tracking of processed videos and analytics
- **Modular Architecture**: Clean, maintainable codebase with separated concerns
- **Resource Management**: Efficient memory and GPU usage optimization

## 🏗️ Architecture

The system has been completely refactored into a modular architecture:

```
src/
├── config/           # Configuration management
├── integrations/     # External API clients (Reddit, YouTube, AI)
├── processing/       # Video processing pipeline
├── database/         # Database operations and analytics
└── orchestrator.py   # Main workflow coordination
```

### Key Components

- **ConfigManager**: Centralized configuration from YAML, environment variables, and defaults
- **RedditClient**: Reddit API integration with content filtering
- **YouTubeClient**: YouTube API integration with upload management
- **AIClient**: Google Gemini integration for video analysis
- **VideoProcessor**: Modular video processing with effects and enhancements
- **DatabaseManager**: SQLite database with comprehensive tracking and analytics

## 📋 Prerequisites

- Python 3.8+
- FFmpeg installed and accessible in PATH
- Reddit API credentials
- Google Cloud project with YouTube Data API v3 enabled
- Google Gemini API key
- At least 4GB RAM (8GB+ recommended)
- GPU optional but recommended for faster processing

## 🔧 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd youtube-video-generation
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up configuration:**
   ```bash
   # Copy the example environment file
   cp secrets.example .env
   
   # Edit .env with your credentials
   nano .env
   ```

4. **Configure API credentials:**

   **Reddit API:**
   - Go to https://www.reddit.com/prefs/apps
   - Create a new application (script type)
   - Add client ID and secret to `.env`

   **Google/YouTube API:**
   - Create a project in Google Cloud Console
   - Enable YouTube Data API v3
   - Create credentials (OAuth 2.0 client ID)
   - Download the JSON file and update path in `.env`

   **Gemini API:**
   - Get API key from https://aistudio.google.com/app/apikey
   - Add to `.env`

5. **Authenticate with YouTube:**
   ```bash
   python auth_youtube.py
   ```

6. **Test the installation:**
   ```bash
   python main.py --dry-run --max-videos 1
   ```

## 🚀 Usage

### Basic Usage

Run the system with default settings:
```bash
python main.py
```

### Command Line Options

```bash
python main.py --help

Options:
  --config PATH         Path to custom configuration file
  --max-videos INT      Maximum number of videos to process (default: 5)
  --subreddits LIST     Specific subreddits to target
  --dry-run            Process videos but don't upload to YouTube
  --debug              Enable debug logging
```

### Examples

```bash
# Process 3 videos in dry-run mode
python main.py --max-videos 3 --dry-run

# Target specific subreddits
python main.py --subreddits oddlysatisfying BeAmazed --max-videos 2

# Enable debug logging
python main.py --debug

# Use custom configuration
python main.py --config my_config.yaml
```

## ⚙️ Configuration

The system uses a hierarchical configuration system:

1. **Default values** (in `src/config/settings.py`)
2. **config.yaml** (project configuration)
3. **Environment variables** (`.env` file)

### Key Configuration Sections

**Video Processing:**
```yaml
video:
  target_duration: 59
  target_resolution: [1080, 1920]
  target_fps: 30
  chunk_size: 30
  max_memory_usage: 0.8
```

**Content Filtering:**
```yaml
content:
  max_reddit_posts_to_fetch: 10
  curated_subreddits: [...]
  forbidden_words: [...]
  unsuitable_content_types: [...]
```

**AI Analysis:**
```yaml
apis:
  gemini:
    model_id: 'gemini-2.0-flash'
    max_frames: 20
    timeout: 30
```

## 🗄️ Database

The system uses SQLite to track:

- **uploads**: Video processing and upload records
- **processing_history**: Detailed step-by-step processing logs
- **analytics**: Daily statistics and performance metrics

### Database Operations

```python
from src.database.db_manager import get_db_manager

db = get_db_manager()

# Check if video was already processed
if db.is_video_processed(reddit_url):
    print("Already processed")

# Get processing statistics
stats = db.get_processing_stats(days=30)
print(f"Success rate: {stats['success_rate']:.1f}%")

# Export data for analysis
db.export_data(Path('backup.json'))
```

## 🔍 Monitoring and Analytics

### Built-in Analytics

The system tracks:
- Processing success rates
- Average processing times
- Popular subreddits
- Error frequencies
- Resource usage patterns

### Logging

Comprehensive logging with configurable levels:
```yaml
logging:
  level: 'INFO'
  enable_file_logging: true
  log_file: 'video_generation.log'
  component_levels:
    'src.integrations.youtube_client': 'DEBUG'
```

## 🛠️ Development

### Project Structure

```
├── main.py                 # Main entry point
├── config.yaml            # Configuration file
├── requirements.txt        # Python dependencies
├── src/
│   ├── config/            # Configuration management
│   ├── integrations/      # External API clients
│   ├── processing/        # Video processing
│   ├── database/          # Database operations
│   └── orchestrator.py    # Main workflow
├── fonts/                 # Font files for text overlays
├── music/                 # Background music files
├── sound_effects/         # Sound effect files
└── temp_processing/       # Temporary processing files
```

### Adding New Features

1. **New Video Effect:**
   - Add method to `VideoEffects` class in `src/processing/video_processor.py`
   - Update configuration schema if needed
   - Add to processing pipeline

2. **New API Integration:**
   - Create new client in `src/integrations/`
   - Follow existing patterns for error handling and configuration
   - Add to orchestrator workflow

3. **New Configuration Option:**
   - Add to appropriate dataclass in `src/config/settings.py`
   - Update `config.yaml` with default value
   - Document in README

### Testing

```bash
# Run with dry-run mode for testing
python main.py --dry-run --debug

# Test specific components
python -m src.integrations.reddit_client
python -m src.integrations.youtube_client
```

## 🚨 Troubleshooting

### Common Issues

**"Reddit client not connected"**
- Check Reddit API credentials in `.env`
- Verify client ID and secret are correct

**"YouTube client not authenticated"**
- Run `python auth_youtube.py` to authenticate
- Check Google client secrets file path
- Verify YouTube API is enabled in Google Cloud Console

**"Gemini API key not set"**
- Get API key from Google AI Studio
- Add to `.env` file as `GEMINI_API_KEY`

**"FFmpeg not found"**
- Install FFmpeg: https://ffmpeg.org/download.html
- Ensure it's in your system PATH

**Memory errors during processing**
- Reduce `max_videos` parameter
- Adjust `max_memory_usage` in config
- Close other applications to free RAM

### Log Analysis

Check `video_generation.log` for detailed error information:
```bash
tail -f video_generation.log
```

## 📊 Performance Optimization

### GPU Acceleration
- Install NVIDIA drivers for GPU encoding
- System automatically detects and uses GPU when available

### Memory Management
- Videos processed in chunks to manage memory
- Automatic cleanup of temporary files
- Resource tracking and garbage collection

### Rate Limiting
- Built-in delays between API calls
- Configurable rate limiting for all services

## 🔒 Security and Privacy

- All credentials stored in environment variables
- No hardcoded API keys
- Automatic content filtering for safety
- GDPR-compliant data handling

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For issues and questions:
- Check the troubleshooting section above
- Review log files for error details
- Open an issue on GitHub with:
  - Error message/logs
  - Configuration (sanitized)
  - Steps to reproduce

## 🔄 Migration from v1.x

If upgrading from the old monolithic system:

1. **Backup your data:**
   ```bash
   cp uploaded_videos.db uploaded_videos.db.backup
   ```

2. **Update configuration:**
   - Review new `config.yaml` format
   - Update `.env` file with new variables

3. **Test in dry-run mode:**
   ```bash
   python main.py --dry-run --max-videos 1
   ```

4. **Database migration:**
   - Automatic schema migration on first run
   - No data loss, new columns added automatically

## 📈 Roadmap

- [ ] TTS integration for narrative audio
- [ ] Advanced thumbnail generation
- [ ] Multi-language support
- [ ] Real-time processing dashboard
- [ ] Advanced analytics and reporting
- [ ] Cloud deployment options
- [ ] API for external integrations