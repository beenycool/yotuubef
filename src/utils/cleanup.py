"""
Cleanup utilities for deleting temporary files, results, and logs.
"""

import logging
import shutil
from pathlib import Path

from src.config.settings import get_config

logger = logging.getLogger(__name__)
config = get_config()

def clear_directory(directory: Path, file_pattern: str = "*"):
    """
    Deletes all files matching a pattern within a specified directory.
    """
    if not directory.exists() or not directory.is_dir():
        logger.warning(f"Cleanup skipped: Directory not found at '{directory}'")
        return

    logger.info(f"Clearing files from '{directory}' matching pattern '{file_pattern}'...")
    files_deleted = 0
    for item in directory.glob(file_pattern):
        try:
            if item.is_file():
                item.unlink()
                logger.debug(f"Deleted file: {item}")
                files_deleted += 1
            elif item.is_dir():
                shutil.rmtree(item)
                logger.debug(f"Deleted directory: {item}")
                files_deleted += 1
        except Exception as e:
            logger.error(f"Failed to delete {item}: {e}")

    logger.info(f"Deleted {files_deleted} items from '{directory}'.")

def clear_temp_files():
    """Clears all files and subdirectories from the temporary data directory."""
    clear_directory(config.paths.temp_dir)

def clear_results():
    """Clears all .json files from the results directory."""
    results_dir = config.paths.base_dir / "data" / "results"
    clear_directory(results_dir, file_pattern="*.json")

def clear_logs():
    """Clears the main log file."""
    logs_dir = config.paths.base_dir / "data" / "logs"
    log_file = logs_dir / "youtube_generator.log"
    if log_file.exists():
        try:
            with open(log_file, 'w') as f:
                f.truncate(0)
            logger.info(f"Cleared log file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to clear log file {log_file}: {e}")
    else:
        logger.warning(f"Log file not found at '{log_file}', skipping cleanup.") 