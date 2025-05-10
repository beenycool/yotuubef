import subprocess
import sys

if __name__ == "__main__":
    try:
        import moviepy.editor
        print("moviepy is already installed.")
    except ImportError:
        print("moviepy not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy"])
        print("moviepy installed successfully.")

    # Now run script.py
    print("Running script.py...")
    subprocess.run([sys.executable, "script.py"]) 