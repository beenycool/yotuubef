#!/usr/bin/env python3
"""
Install spotify_dl for music downloading functionality
"""

import subprocess
import sys
import os

def install_spotify_dl():
    """Install spotify_dl package"""
    try:
        print("Installing spotify_dl...")
        
        # Install spotify_dl using pip
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "spotify_dl"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ spotify_dl installed successfully!")
            
            # Test the installation
            test_result = subprocess.run(["spotify_dl", "--version"], 
                                       capture_output=True, text=True)
            if test_result.returncode == 0:
                print("✅ spotify_dl is working correctly!")
                print(f"Version: {test_result.stdout.strip()}")
            else:
                print("⚠️ spotify_dl installed but may not be in PATH")
                
        else:
            print("❌ Failed to install spotify_dl")
            print(f"Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Installation failed: {e}")

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        # Check if ffmpeg is available
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ ffmpeg is available")
        else:
            print("⚠️ ffmpeg not found - some features may not work")
            print("Please install ffmpeg: https://ffmpeg.org/download.html")
            
    except FileNotFoundError:
        print("⚠️ ffmpeg not found - some features may not work")
        print("Please install ffmpeg: https://ffmpeg.org/download.html")

if __name__ == "__main__":
    print("YouTube Shorts Generator - Spotify DL Installer")
    print("=" * 50)
    
    check_dependencies()
    install_spotify_dl()
    
    print("\n" + "=" * 50)
    print("Installation complete!")
    print("You can now run the main script: python main.py")