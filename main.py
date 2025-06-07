#!/usr/bin/env python3
"""
YouTube Shorts Video Generation System - Main Entry Point

A complete refactored system for automatically generating YouTube Shorts
from Reddit content with AI-powered analysis and enhancement.

Version: 2.0.0
"""

import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the main orchestrator
from src.orchestrator import main

if __name__ == '__main__':
    exit(main())