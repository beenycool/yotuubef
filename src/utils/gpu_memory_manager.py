"""
GPU Memory Management Utilities for AI Models
Handles GPU memory optimization, monitoring, and efficient model loading
"""

import logging
import gc
from typing import Optional, Dict, Any, Tuple
import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class GPUMemoryManager:
    """
    Manages GPU memory allocation and optimization for AI models
    """
    
    def __init__(self, max_vram_usage: float = 0.85):
        """
        Initialize GPU memory manager
        
        Args:
            max_vram_usage: Maximum VRAM usage ratio (0.0 to 1.0)
        """
        self.logger = logging.getLogger(__name__)
        self.max_vram_usage = max_vram_usage
        self.device = None
        self.vram_limit_mb = None
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
            except Exception as e:
                self.logger.warning(f"Failed to initialize NVML: {e}")
                self.nvml_initialized = False
        else:
            self.nvml_initialized = False
        
        self._initialize_device()
    
    def _initialize_device(self) -> None:
        """Initialize CUDA device and check VRAM availability"""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, using CPU only")
            self.device = "cpu"
            return
        
        if not torch.cuda.is_available():
            self.logger.info("CUDA not available, using CPU")
            self.device = "cpu"
            return
        
        # Get GPU info
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            self.logger.info("No CUDA devices found, using CPU")
            self.device = "cpu"
            return
        
        # Use first GPU by default
        self.device = "cuda:0"
        
        # Get VRAM info
        vram_info = self.get_vram_info()
        if vram_info:
            total_vram_gb = vram_info['total'] / 1024**3
            self.vram_limit_mb = int(vram_info['total'] * self.max_vram_usage / 1024**2)
            
            self.logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"Total VRAM: {total_vram_gb:.1f}GB")
            self.logger.info(f"VRAM limit set to: {self.vram_limit_mb}MB ({self.max_vram_usage*100:.0f}%)")
            
            # Check if we have sufficient VRAM for typical models
            if total_vram_gb < 4.0:
                self.logger.warning(f"Low VRAM detected ({total_vram_gb:.1f}GB). Consider using CPU or optimized models.")
    
    def get_vram_info(self) -> Optional[Dict[str, int]]:
        """
        Get current VRAM usage information
        
        Returns:
            Dictionary with 'total', 'used', 'free' VRAM in bytes, or None if unavailable
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None
        
        try:
            if self.nvml_initialized:
                # Use NVML for accurate VRAM info
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return {
                    'total': info.total,
                    'used': info.used,
                    'free': info.free
                }
            else:
                # Fallback to PyTorch CUDA memory info
                total = torch.cuda.get_device_properties(0).total_memory
                allocated = torch.cuda.memory_allocated(0)
                cached = torch.cuda.memory_reserved(0)
                return {
                    'total': total,
                    'used': allocated,
                    'free': total - cached
                }
        except Exception as e:
            self.logger.warning(f"Failed to get VRAM info: {e}")
            return None
    
    def check_vram_available(self, required_mb: int) -> bool:
        """
        Check if sufficient VRAM is available for model loading
        
        Args:
            required_mb: Required VRAM in MB
            
        Returns:
            True if sufficient VRAM available
        """
        if self.device == "cpu":
            return False
        
        vram_info = self.get_vram_info()
        if not vram_info:
            return False
        
        free_mb = vram_info['free'] / 1024**2
        return free_mb >= required_mb
    
    def get_optimal_device(self, model_size_mb: int = 1000) -> str:
        """
        Get optimal device for model loading based on VRAM availability
        
        Args:
            model_size_mb: Estimated model size in MB
            
        Returns:
            Device string ('cuda:0' or 'cpu')
        """
        if self.device == "cpu":
            return "cpu"
        
        if self.check_vram_available(model_size_mb):
            return self.device
        else:
            self.logger.warning(f"Insufficient VRAM for {model_size_mb}MB model, falling back to CPU")
            return "cpu"
    
    def optimize_model_for_inference(self, model: Any) -> Any:
        """
        Optimize model for inference to reduce memory usage
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model
        """
        if not TORCH_AVAILABLE:
            return model
        
        try:
            # Enable evaluation mode
            model.eval()
            
            # Disable gradient computation
            for param in model.parameters():
                param.requires_grad = False
            
            # Enable half precision if on GPU and supported
            if hasattr(model, 'half') and next(model.parameters()).device.type == 'cuda':
                try:
                    model = model.half()
                    self.logger.debug("Enabled half precision for model")
                except Exception as e:
                    self.logger.warning(f"Could not enable half precision: {e}")
            
            return model
            
        except Exception as e:
            self.logger.warning(f"Could not optimize model: {e}")
            return model
    
    def clear_gpu_cache(self) -> None:
        """Clear GPU memory cache"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            self.logger.debug("Cleared GPU memory cache")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive memory usage summary
        
        Returns:
            Dictionary with memory usage information
        """
        summary = {
            'device': self.device,
            'torch_available': TORCH_AVAILABLE,
            'cuda_available': torch.cuda.is_available() if TORCH_AVAILABLE else False
        }
        
        # System RAM info
        try:
            ram = psutil.virtual_memory()
            summary['system_ram'] = {
                'total_gb': ram.total / 1024**3,
                'used_gb': ram.used / 1024**3,
                'available_gb': ram.available / 1024**3,
                'percent_used': ram.percent
            }
        except Exception as e:
            summary['system_ram'] = {'error': str(e)}
        
        # VRAM info
        vram_info = self.get_vram_info()
        if vram_info:
            summary['vram'] = {
                'available': True,
                'total_gb': vram_info['total'] / 1024**3,
                'used_gb': vram_info['used'] / 1024**3,
                'free_gb': vram_info['free'] / 1024**3,
                'percent_used': (vram_info['used'] / vram_info['total']) * 100,
                'limit_gb': self.vram_limit_mb / 1024 if self.vram_limit_mb else None
            }
        else:
            summary['vram'] = {'available': False}
        
        return summary
    
    def log_memory_status(self, context: str = "Memory Status") -> None:
        """Log current memory status"""
        summary = self.get_memory_summary()
        
        self.logger.info(f"=== {context} ===")
        self.logger.info(f"Device: {summary['device']}")
        
        if 'system_ram' in summary and 'total_gb' in summary['system_ram']:
            ram = summary['system_ram']
            self.logger.info(f"System RAM: {ram['used_gb']:.1f}GB / {ram['total_gb']:.1f}GB ({ram['percent_used']:.1f}%)")
        
        if summary['vram']['available']:
            vram = summary['vram']
            self.logger.info(f"VRAM: {vram['used_gb']:.1f}GB / {vram['total_gb']:.1f}GB ({vram['percent_used']:.1f}%)")
        else:
            self.logger.info("VRAM: Not available")


# Global memory manager instance
_memory_manager = None

def get_memory_manager() -> GPUMemoryManager:
    """Get global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = GPUMemoryManager()
    return _memory_manager