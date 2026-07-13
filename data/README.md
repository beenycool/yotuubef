# Data Directory Structure

This directory contains all data files organized by type:

## Consolidated Database
- `youtube_shorts.db` - Single SQLite database containing all tables (uploads, local_artifacts, processing_history, video_metrics, enhancement_tracking, performance_snapshots, ab_test_results)

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