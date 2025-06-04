import shutil
import os
from pathlib import Path

def clean_temp_files():
    """Clean up temporary processing files"""
    temp_dir = Path("temp_processing")
    
    if temp_dir.exists():
        print(f"Cleaning temp directory: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
            temp_dir.mkdir(exist_ok=True)
            print("Successfully cleaned temp directory")
        except Exception as e:
            print(f"Error cleaning temp directory: {e}")
    else:
        print("No temp directory found")

if __name__ == "__main__":
    clean_temp_files()