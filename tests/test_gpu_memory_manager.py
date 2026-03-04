import pytest
import sys
import gc
from unittest.mock import MagicMock, patch

# Need to mock torch and pynvml before importing the module under test
# because it uses try/except block on module level

import src.utils.gpu_memory_manager as gmm
from src.utils.gpu_memory_manager import GPUMemoryManager, get_memory_manager

@pytest.fixture
def no_torch(monkeypatch):
    monkeypatch.setattr(gmm, 'TORCH_AVAILABLE', False)
    monkeypatch.setattr(gmm, 'PYNVML_AVAILABLE', False)
    yield

@pytest.fixture
def with_torch_no_cuda(monkeypatch):
    monkeypatch.setattr(gmm, 'TORCH_AVAILABLE', True)
    monkeypatch.setattr(gmm, 'PYNVML_AVAILABLE', False)
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setattr(gmm, 'torch', mock_torch, raising=False)
    yield mock_torch

@pytest.fixture
def with_cuda(monkeypatch):
    monkeypatch.setattr(gmm, 'TORCH_AVAILABLE', True)
    monkeypatch.setattr(gmm, 'PYNVML_AVAILABLE', False)
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mock_torch.cuda.get_device_name.return_value = "Mock GPU"

    mock_prop = MagicMock()
    mock_prop.total_memory = 8 * 1024**3 # 8 GB
    mock_torch.cuda.get_device_properties.return_value = mock_prop
    mock_torch.cuda.memory_allocated.return_value = 2 * 1024**3 # 2 GB
    mock_torch.cuda.memory_reserved.return_value = 3 * 1024**3 # 3 GB

    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setattr(gmm, 'torch', mock_torch, raising=False)
    yield mock_torch

@pytest.fixture
def with_cuda_pynvml(with_cuda, monkeypatch):
    monkeypatch.setattr(gmm, 'PYNVML_AVAILABLE', True)
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit = MagicMock()

    mock_info = MagicMock()
    mock_info.total = 8 * 1024**3
    mock_info.used = 2 * 1024**3
    mock_info.free = 6 * 1024**3
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_info

    monkeypatch.setitem(sys.modules, "pynvml", mock_pynvml)
    monkeypatch.setattr(gmm, 'pynvml', mock_pynvml, raising=False)
    yield with_cuda, mock_pynvml


def test_init_no_torch(no_torch):
    manager = GPUMemoryManager()
    assert manager.device == "cpu"
    assert manager.vram_limit_mb is None

def test_init_torch_no_cuda(with_torch_no_cuda):
    manager = GPUMemoryManager()
    assert manager.device == "cpu"
    assert manager.vram_limit_mb is None

def test_init_with_cuda(with_cuda):
    manager = GPUMemoryManager(max_vram_usage=0.5)
    assert manager.device == "cuda:0"
    assert manager.vram_limit_mb == int(8 * 1024**3 * 0.5 / 1024**2)

def test_init_with_cuda_low_vram(monkeypatch):
    monkeypatch.setattr(gmm, 'TORCH_AVAILABLE', True)
    monkeypatch.setattr(gmm, 'PYNVML_AVAILABLE', False)
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1

    mock_prop = MagicMock()
    mock_prop.total_memory = 2 * 1024**3 # 2 GB
    mock_torch.cuda.get_device_properties.return_value = mock_prop

    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setattr(gmm, 'torch', mock_torch, raising=False)

    manager = GPUMemoryManager()
    # It should still set the device to cuda:0, just log a warning
    assert manager.device == "cuda:0"

def test_init_pynvml_fails(with_cuda, monkeypatch):
    monkeypatch.setattr(gmm, 'PYNVML_AVAILABLE', True)
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.side_effect = Exception("NVML Error")
    monkeypatch.setitem(sys.modules, "pynvml", mock_pynvml)
    monkeypatch.setattr(gmm, 'pynvml', mock_pynvml, raising=False)

    manager = GPUMemoryManager()
    assert manager.nvml_initialized is False
    assert manager.device == "cuda:0"


def test_get_vram_info_no_cuda(with_torch_no_cuda):
    manager = GPUMemoryManager()
    assert manager.get_vram_info() is None

def test_get_vram_info_cuda_no_nvml(with_cuda):
    manager = GPUMemoryManager()
    info = manager.get_vram_info()
    assert info is not None
    assert info['total'] == 8 * 1024**3
    assert info['used'] == 2 * 1024**3
    assert info['free'] == 8 * 1024**3 - 3 * 1024**3 # total - reserved

def test_get_vram_info_cuda_with_nvml(with_cuda_pynvml):
    torch_mock, pynvml_mock = with_cuda_pynvml
    manager = GPUMemoryManager()
    info = manager.get_vram_info()
    assert info is not None
    assert info['total'] == 8 * 1024**3
    assert info['used'] == 2 * 1024**3
    assert info['free'] == 6 * 1024**3

def test_check_vram_available_cpu(with_torch_no_cuda):
    manager = GPUMemoryManager()
    assert manager.check_vram_available(100) is False

def test_check_vram_available_cuda(with_cuda):
    manager = GPUMemoryManager()
    # Free is 5GB = 5120MB
    assert manager.check_vram_available(5000) is True
    assert manager.check_vram_available(6000) is False

def test_get_optimal_device_cpu(with_torch_no_cuda):
    manager = GPUMemoryManager()
    assert manager.get_optimal_device(100) == "cpu"

def test_get_optimal_device_cuda(with_cuda):
    manager = GPUMemoryManager()
    # Free is 5GB = 5120MB
    assert manager.get_optimal_device(5000) == "cuda:0"
    assert manager.get_optimal_device(6000) == "cpu"

def test_optimize_model_for_inference_no_torch(no_torch):
    manager = GPUMemoryManager()
    mock_model = MagicMock()
    optimized = manager.optimize_model_for_inference(mock_model)
    assert optimized is mock_model

def test_optimize_model_for_inference_cpu(with_torch_no_cuda):
    manager = GPUMemoryManager()
    mock_model = MagicMock()
    mock_param = MagicMock()
    mock_param.requires_grad = True
    mock_param.device.type = 'cpu'

    mock_model.parameters = lambda: iter([mock_param])

    optimized = manager.optimize_model_for_inference(mock_model)
    assert optimized is mock_model
    mock_model.eval.assert_called_once()
    assert mock_param.requires_grad is False
    # half() should not be called since device is cpu

def test_optimize_model_for_inference_cuda(with_cuda):
    manager = GPUMemoryManager()
    mock_model = MagicMock()
    mock_param = MagicMock()
    mock_param.requires_grad = True
    mock_param.device.type = 'cuda'

    mock_model.parameters = lambda: iter([mock_param])
    mock_half_model = MagicMock()
    mock_model.half.return_value = mock_half_model

    optimized = manager.optimize_model_for_inference(mock_model)
    assert optimized is mock_half_model
    mock_model.eval.assert_called_once()
    assert mock_param.requires_grad is False
    mock_model.half.assert_called_once()

def test_optimize_model_for_inference_half_fails(with_cuda):
    manager = GPUMemoryManager()
    mock_model = MagicMock()
    mock_param = MagicMock()
    mock_param.requires_grad = True
    mock_param.device.type = 'cuda'

    mock_model.parameters = lambda: iter([mock_param])
    mock_model.half.side_effect = Exception("Half precision failed")

    optimized = manager.optimize_model_for_inference(mock_model)
    # the original model should be returned if half fails
    assert optimized is mock_model

def test_clear_gpu_cache_no_torch(no_torch):
    manager = GPUMemoryManager()
    manager.clear_gpu_cache() # should not raise anything

def test_clear_gpu_cache_with_cuda(with_cuda):
    mock_torch = with_cuda
    manager = GPUMemoryManager()

    with patch('src.utils.gpu_memory_manager.gc.collect') as mock_gc:
        manager.clear_gpu_cache()
        mock_torch.cuda.empty_cache.assert_called_once()
        mock_gc.assert_called_once()

def test_get_memory_summary_no_torch(no_torch):
    manager = GPUMemoryManager()

    with patch('src.utils.gpu_memory_manager.psutil.virtual_memory') as mock_vm:
        mock_vm.return_value = MagicMock(total=16 * 1024**3, used=8 * 1024**3, available=8 * 1024**3, percent=50.0)
        summary = manager.get_memory_summary()

        assert summary['device'] == "cpu"
        assert summary['torch_available'] is False
        assert summary['cuda_available'] is False
        assert summary['system_ram']['total_gb'] == 16.0
        assert summary['system_ram']['used_gb'] == 8.0
        assert summary['vram']['available'] is False

def test_get_memory_summary_psutil_fails(no_torch):
    manager = GPUMemoryManager()

    with patch('src.utils.gpu_memory_manager.psutil.virtual_memory') as mock_vm:
        mock_vm.side_effect = Exception("psutil failed")
        summary = manager.get_memory_summary()

        assert 'error' in summary['system_ram']
        assert summary['system_ram']['error'] == "psutil failed"

def test_get_memory_summary_cuda(with_cuda):
    manager = GPUMemoryManager()

    with patch('src.utils.gpu_memory_manager.psutil.virtual_memory') as mock_vm:
        mock_vm.return_value = MagicMock(total=16 * 1024**3, used=8 * 1024**3, available=8 * 1024**3, percent=50.0)
        summary = manager.get_memory_summary()

        assert summary['device'] == "cuda:0"
        assert summary['torch_available'] is True
        assert summary['cuda_available'] is True
        assert summary['vram']['available'] is True
        assert summary['vram']['total_gb'] == 8.0
        assert summary['vram']['used_gb'] == 2.0
        # 5.0 GB free (8 total - 3 reserved)
        assert summary['vram']['free_gb'] == 5.0
        assert summary['vram']['percent_used'] == (2 / 8) * 100

def test_log_memory_status_cuda(with_cuda):
    manager = GPUMemoryManager()

    with patch('src.utils.gpu_memory_manager.psutil.virtual_memory') as mock_vm:
        mock_vm.return_value = MagicMock(total=16 * 1024**3, used=8 * 1024**3, available=8 * 1024**3, percent=50.0)

        with patch.object(manager.logger, 'info') as mock_logger:
            manager.log_memory_status("Test Context")

            # Verify logger was called
            assert mock_logger.call_count >= 3
            mock_logger.assert_any_call("=== Test Context ===")
            mock_logger.assert_any_call("Device: cuda:0")

def test_log_memory_status_no_vram(no_torch):
    manager = GPUMemoryManager()

    with patch('src.utils.gpu_memory_manager.psutil.virtual_memory') as mock_vm:
        mock_vm.return_value = MagicMock(total=16 * 1024**3, used=8 * 1024**3, available=8 * 1024**3, percent=50.0)

        with patch.object(manager.logger, 'info') as mock_logger:
            manager.log_memory_status("Test Context")

            mock_logger.assert_any_call("VRAM: Not available")

def test_get_memory_manager():
    # Reset global singleton to test it
    gmm._memory_manager = None
    manager1 = get_memory_manager()
    manager2 = get_memory_manager()

    assert manager1 is not None
    assert manager1 is manager2
