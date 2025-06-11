# Data Directory Structure

This directory contains all data files organized by type:

## `/databases/`
- `engagement_metrics.db` - Video performance and engagement metrics
- `uploaded_videos.db` - Track uploaded videos and prevent duplicates

## `/logs/`
- `youtube_generator.log` - Main application logs
- Other application log files

## `/results/`
- `results_find_*.json` - Results from video finding operations
- `results_single_*.json` - Single video processing results
- `results_batch_*.json` - Batch processing results
- `optimization_*.json` - System optimization results

## `/temp/`
- Temporary video processing files
- Downloaded videos before processing
- Work-in-progress files

Note: Most files in these directories are gitignored for privacy and storage reasons. 