"""
Script to install GPU support for the YouTube Shorts generator
"""

import subprocess
import sys
import platform

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available"""
    success, stdout, stderr = run_command("nvidia-smi")
    if success:
        print("✅ NVIDIA GPU detected")
        print(stdout.split('\n')[2:4])  # Show GPU info lines
        return True
    else:
        print("❌ NVIDIA GPU not detected or nvidia-smi not available")
        return False

def install_pytorch_cuda():
    """Install PyTorch with CUDA support"""
    print("Installing PyTorch with CUDA support...")
    
    # Uninstall existing PyTorch
    print("Uninstalling existing PyTorch...")
    run_command("pip uninstall torch torchvision torchaudio -y")
    
    # Install CUDA-enabled PyTorch
    print("Installing PyTorch with CUDA 12.1 support...")
    success, stdout, stderr = run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    )
    
    if success:
        print("✅ PyTorch with CUDA installed successfully")
    else:
        print("❌ Failed to install PyTorch with CUDA")
        print(f"Error: {stderr}")
        return False
    
    return True

def verify_installation():
    """Verify that PyTorch can detect CUDA"""
    print("Verifying PyTorch CUDA installation...")
    
    test_script = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    print("CUDA not available in PyTorch")
"""
    
    success, stdout, stderr = run_command(f'python -c "{test_script}"')
    
    if success and "CUDA available: True" in stdout:
        print("✅ PyTorch CUDA verification successful")
        print(stdout)
        return True
    else:
        print("❌ PyTorch CUDA verification failed")
        print(f"Output: {stdout}")
        print(f"Error: {stderr}")
        return False

def install_additional_packages():
    """Install additional GPU-related packages"""
    packages = [
        "pynvml",  # GPU monitoring
        "accelerate",  # Model optimization
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        success, stdout, stderr = run_command(f"pip install {package}")
        if success:
            print(f"✅ {package} installed")
        else:
            print(f"❌ Failed to install {package}: {stderr}")

def main():
    print("=== GPU Support Installation Script ===")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    # Check for NVIDIA GPU
    if not check_nvidia_gpu():
        print("\n⚠️  No NVIDIA GPU detected. GPU acceleration will not be available.")
        return
    
    # Install PyTorch with CUDA
    if not install_pytorch_cuda():
        return
    
    # Verify installation
    if not verify_installation():
        return
    
    # Install additional packages
    install_additional_packages()
    
    print("\n🎉 GPU support installation completed!")
    print("\nNext steps:")
    print("1. Test GPU optimization: python tests/test_gpu_memory_optimization.py")
    print("2. Run your application and check for GPU usage in logs")

if __name__ == "__main__":
    main()