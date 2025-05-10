import torch
import logging
import sys
from transformers import pipeline
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def test_transformers_cuda():
    """Test if transformers can use the GPU for inference"""
    logging.info(f"PyTorch version: {torch.__version__}")
    
    # Test 1: Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    logging.info(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        logging.error("CUDA is not available. Check NVIDIA drivers and PyTorch installation.")
        return False
    
    # Set current device and get current device
    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    logging.info(f"Using GPU: {device_name}")
    
    # Test 2: Try running a simple text classification pipeline on GPU vs CPU
    try:
        # First on CPU for comparison
        logging.info("Running transformers pipeline on CPU...")
        cpu_start = time.time()
        cpu_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
        cpu_result = cpu_classifier("This is a test to see if transformers works on GPU")
        cpu_time = time.time() - cpu_start
        logging.info(f"CPU result: {cpu_result}")
        logging.info(f"CPU time: {cpu_time:.2f} seconds")
        
        # Now on GPU
        logging.info("Running transformers pipeline on GPU...")
        gpu_start = time.time()
        gpu_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)
        gpu_result = gpu_classifier("This is a test to see if transformers works on GPU")
        gpu_time = time.time() - gpu_start
        logging.info(f"GPU result: {gpu_result}")
        logging.info(f"GPU time: {gpu_time:.2f} seconds")
        
        # Compare performance
        if gpu_time < cpu_time:
            logging.info(f"GPU is {cpu_time/gpu_time:.2f}x faster than CPU!")
        else:
            logging.warning(f"GPU is not faster than CPU. This might be due to initialization overhead or other issues.")
        
        logging.info("Transformers GPU test PASSED - GPU is working correctly!")
        return True
    except Exception as e:
        logging.error(f"Error running transformers on GPU: {e}")
        return False

if __name__ == "__main__":
    test_transformers_cuda() 