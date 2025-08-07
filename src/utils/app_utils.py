"""
Application setup and initialization utilities.
"""

import logging
import shutil
from pathlib import Path
from src.config.settings import get_config, setup_logging


def setup_application():
    """Initialize basic configuration and logging"""
    get_config()
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("YouTube Shorts Generator initialized")


def clear_temp_files():
    """Clear temporary files - maintains original main.py behavior"""
    temp_dir = Path("temp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        temp_dir.mkdir(exist_ok=True)


def clear_results():
    """Clear results files - maintains original main.py behavior"""
    results_dir = Path("data/results")
    if results_dir.exists():
        for file in results_dir.glob("*"):
            if file.is_file():
                file.unlink()


def clear_logs():
    """Clear log files - maintains original main.py behavior"""
    logs_dir = Path("logs")
    if logs_dir.exists():
        for file in logs_dir.glob("*.log"):
            file.unlink()
        for file in logs_dir.glob("*.jsonl"):
            file.unlink()