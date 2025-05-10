import torch
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def test_pytorch_cuda():
    """Test if PyTorch can access the GPU"""
    logging.info(f"PyTorch version: {torch.__version__}")
    
    # Test 1: Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    logging.info(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        logging.error("CUDA is not available. Check NVIDIA drivers and PyTorch installation.")
        return False
    
    # Test 2: Check CUDA device information
    device_count = torch.cuda.device_count()
    logging.info(f"CUDA device count: {device_count}")
    
    if device_count == 0:
        logging.error("No CUDA devices found despite CUDA being available.")
        return False
    
    # Test 3: Get device properties
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        device_capability = torch.cuda.get_device_capability(i)
        logging.info(f"Device {i}: {device_name}, Compute Capability: {device_capability}")
    
    # Test 4: Set current device and get current device
    torch.cuda.set_device(0)
    current_device = torch.cuda.current_device()
    logging.info(f"Current CUDA device: {current_device}")
    
    # Test 5: Run a simple operation on GPU to verify it works
    try:
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        z = torch.matmul(x, y)
        logging.info(f"Matrix multiplication result shape: {z.shape}")
        logging.info("PyTorch CUDA test PASSED - GPU is working correctly!")
        return True
    except Exception as e:
        logging.error(f"Error running operations on GPU: {e}")
        return False

if __name__ == "__main__":
    test_pytorch_cuda() 