# Reddit Video to YouTube Shorts - Google Colab T4 Version

This repository contains a script that adapts the video processing pipeline to run on Google Colab with T4 GPU acceleration. This allows you to process Reddit videos and turn them into YouTube Shorts without needing powerful hardware on your own machine.

## How to Use

1. Go to [Google Colab](https://colab.research.google.com/) and create a new notebook
2. Upload these files to your Colab session:
   - `colab_adapter.py` (the adapter script)
   - `script.py` (your original script)

3. In a code cell in your Colab notebook, run:
   ```python
   !python colab_adapter.py
   ```

4. Follow the prompts to:
   - Upload your Google API credentials JSON file
   - Enter your Reddit API credentials
   - Enter your Gemini API key
   - (Optional) Enter your ElevenLabs API key
   - Choose which subreddits to process
   - Set how many videos to process per subreddit

## What the Adapter Does

The adapter script performs these steps automatically:

1. Installs all required dependencies
2. Sets up proper configurations for Google Colab
3. Creates the video_processor.py module
4. Downloads royalty-free background music
5. Adapts the original script to use T4 GPU acceleration
6. Sets up all necessary directories and files
7. Collects your API credentials
8. Runs the optimized script

## Requirements

- Reddit API credentials (client ID and secret)
- Google Cloud project with YouTube API enabled and OAuth credentials
- Gemini API key
- (Optional) ElevenLabs API key for better TTS

## GPU Optimization

The adapted script automatically uses the T4 GPU in Colab for these operations:
- Video encoding/decoding with NVENC
- GPU-accelerated processing
- TTS generation (if using Dia-1.6B model with torch)

## Troubleshooting

- If you encounter memory issues, try reducing the `max_videos` parameter
- For video processing errors, check that the input videos are valid
- If GPU acceleration isn't working, the script will automatically fall back to CPU processing

## Notes

- Videos will be saved in the Colab `/content/temp_processing` directory
- You can download processed videos from the Colab file browser
- For longer sessions, consider upgrading to Colab Pro or enabling "Keep notebook running when browser tab closes" option 