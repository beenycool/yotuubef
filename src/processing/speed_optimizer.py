"""
Speed optimization utilities for MoviePy encoding.
Automatically detects system capabilities and optimizes encoding settings.
"""

import logging
import subprocess
import psutil
import platform
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class SystemCapabilities:
    """System capabilities for video encoding optimization"""
    cpu_cores: int
    ram_gb: float
    gpu_available: bool
    gpu_memory_gb: float
    nvenc_available: bool
    quicksync_available: bool
    platform: str
    
class SpeedOptimizer:
    """Optimizes MoviePy encoding settings based on system capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.capabilities = self._detect_system_capabilities()
        
    def _detect_system_capabilities(self) -> SystemCapabilities:
        """Detect system capabilities for optimization"""
        try:
            # CPU and RAM
            cpu_cores = psutil.cpu_count()
            ram_gb = psutil.virtual_memory().total / (1024**3)
            
            # GPU detection
            gpu_available = False
            gpu_memory_gb = 0.0
            nvenc_available = False
            quicksync_available = False
            
            try:
                # Try GPUtil first
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_available = True
                    gpu_memory_gb = max(gpu.memoryTotal / 1024 for gpu in gpus)  # Convert MB to GB
                    
                    # Check for NVENC (NVIDIA)
                    for gpu in gpus:
                        if 'nvidia' in gpu.name.lower():
                            nvenc_available = True
                            break
            except ImportError:
                # Fallback: check nvidia-smi
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True, check=True)
                    memory_mb = int(result.stdout.strip().split('\n')[0])
                    gpu_available = True
                    nvenc_available = True
                    gpu_memory_gb = memory_mb / 1024
                except:
                    pass
            
            # Check for Intel QuickSync
            try:
                if 'intel' in platform.processor().lower():
                    quicksync_available = True
            except:
                pass
            
            capabilities = SystemCapabilities(
                cpu_cores=cpu_cores,
                ram_gb=ram_gb,
                gpu_available=gpu_available,
                gpu_memory_gb=gpu_memory_gb,
                nvenc_available=nvenc_available,
                quicksync_available=quicksync_available,
                platform=platform.system()
            )
            
            self.logger.info(f"Detected system: {cpu_cores} cores, {ram_gb:.1f}GB RAM, "
                           f"GPU: {gpu_available} (NVENC: {nvenc_available})")
            
            return capabilities
            
        except Exception as e:
            self.logger.warning(f"Error detecting system capabilities: {e}")
            # Return minimal capabilities
            return SystemCapabilities(
                cpu_cores=4, ram_gb=8.0, gpu_available=False, gpu_memory_gb=0.0,
                nvenc_available=False, quicksync_available=False, platform="Unknown"
            )
    
    def get_optimal_settings(self, 
                           target_quality: str = 'standard',
                           target_speed: str = 'aggressive') -> Dict[str, Any]:
        """Get optimal encoding settings based on system capabilities"""
        
        settings = {
            'codec': 'libx264',
            'preset': 'medium',
            'crf': '23',
            'bitrate': '10M',
            'threads': 4,
            'ffmpeg_params': [],
            'additional_params': {}
        }
        
        # Choose optimal codec based on hardware
        if self.capabilities.nvenc_available:
            settings['codec'] = 'h264_nvenc'
            settings.update(self._get_nvenc_settings(target_quality, target_speed))
        elif self.capabilities.quicksync_available:
            settings['codec'] = 'h264_qsv'
            settings.update(self._get_quicksync_settings(target_quality, target_speed))
        else:
            settings.update(self._get_x264_settings(target_quality, target_speed))
        
        # Optimize thread count
        settings['threads'] = self._get_optimal_threads(target_speed)
        
        # Add speed-specific optimizations
        settings['additional_params'].update(self._get_speed_params(target_speed))
        
        return settings
    
    def _get_nvenc_settings(self, quality: str, speed: str) -> Dict[str, Any]:
        """Get NVENC-specific optimized settings"""
        settings = {}
        
        if speed == 'aggressive':
            settings.update({
                'preset': 'p1',  # Fastest preset
                'crf': '28' if quality == 'speed' else '25',
                'bitrate': '6M' if quality == 'speed' else '8M',
                'ffmpeg_params': [
                    '-rc', 'vbr',
                    '-rc-lookahead', '10',  # Minimal lookahead
                    '-surfaces', '32',  # Reduced surfaces for speed
                    '-bf', '2',  # Fewer B-frames
                    '-gpu', '0',
                    '-delay', '0',  # No delay
                    '-zerolatency', '1'  # Zero latency mode
                ]
            })
        elif speed == 'balanced':
            settings.update({
                'preset': 'p3',
                'crf': '25' if quality == 'speed' else '23',
                'bitrate': '8M' if quality == 'speed' else '10M',
                'ffmpeg_params': [
                    '-rc', 'vbr',
                    '-rc-lookahead', '20',
                    '-surfaces', '64',
                    '-bf', '3',
                    '-gpu', '0'
                ]
            })
        else:  # conservative
            settings.update({
                'preset': 'p5',
                'crf': '23',
                'bitrate': '12M',
                'ffmpeg_params': [
                    '-rc', 'vbr',
                    '-rc-lookahead', '32',
                    '-surfaces', '64',
                    '-bf', '4',
                    '-gpu', '0'
                ]
            })
        
        return settings
    
    def _get_quicksync_settings(self, quality: str, speed: str) -> Dict[str, Any]:
        """Get Intel QuickSync optimized settings"""
        settings = {}
        
        if speed == 'aggressive':
            settings.update({
                'preset': 'veryfast',
                'crf': '28' if quality == 'speed' else '25',
                'bitrate': '6M' if quality == 'speed' else '8M',
                'ffmpeg_params': ['-look_ahead', '0', '-async_depth', '1']
            })
        else:
            settings.update({
                'preset': 'fast',
                'crf': '25' if quality == 'speed' else '23',
                'bitrate': '8M' if quality == 'speed' else '10M',
                'ffmpeg_params': ['-look_ahead', '1', '-async_depth', '4']
            })
        
        return settings
    
    def _get_x264_settings(self, quality: str, speed: str) -> Dict[str, Any]:
        """Get x264 optimized settings"""
        settings = {}
        
        if speed == 'aggressive':
            settings.update({
                'preset': 'ultrafast',
                'crf': '28' if quality == 'speed' else '26',
                'bitrate': '6M' if quality == 'speed' else '8M',
                'ffmpeg_params': [
                    '-tune', 'fastdecode,zerolatency',
                    '-x264-params', 'ref=1:bframes=0:subme=1:me=dia:no-cabac=1:trellis=0:weightp=0'
                ]
            })
        elif speed == 'balanced':
            settings.update({
                'preset': 'fast',
                'crf': '25' if quality == 'speed' else '23',
                'bitrate': '8M' if quality == 'speed' else '10M',
                'ffmpeg_params': [
                    '-tune', 'fastdecode',
                    '-x264-params', 'ref=2:bframes=2:subme=4:me=hex:trellis=1'
                ]
            })
        else:  # conservative
            settings.update({
                'preset': 'medium',
                'crf': '23',
                'bitrate': '12M',
                'ffmpeg_params': ['-tune', 'film']
            })
        
        return settings
    
    def _get_optimal_threads(self, speed: str) -> int:
        """Calculate optimal thread count based on CPU and speed requirements"""
        cpu_cores = self.capabilities.cpu_cores
        
        if speed == 'aggressive':
            # Use all available cores
            return cpu_cores
        elif speed == 'balanced':
            # Leave 1-2 cores free for system
            return max(2, cpu_cores - 2)
        else:  # conservative
            # Use half the cores to avoid system slowdown
            return max(2, cpu_cores // 2)
    
    def _get_speed_params(self, speed: str) -> Dict[str, Any]:
        """Get additional parameters for speed optimization"""
        params = {}
        
        if speed == 'aggressive':
            params.update({
                'temp_audiofile': None,
                'remove_temp': True,
                'audio_bitrate': '128k',  # Lower audio bitrate for speed
            })
        elif speed == 'balanced':
            params.update({
                'temp_audiofile': None,
                'remove_temp': True,
                'audio_bitrate': '192k'
            })
        
        return params
    
    def get_recommended_profile(self) -> str:
        """Get recommended speed optimization profile based on system"""
        if self.capabilities.nvenc_available and self.capabilities.gpu_memory_gb >= 4:
            return 'aggressive'
        elif self.capabilities.cpu_cores >= 8 and self.capabilities.ram_gb >= 16:
            return 'balanced'
        else:
            return 'conservative'


def create_speed_optimizer() -> SpeedOptimizer:
    """Factory function to create a speed optimizer"""
    return SpeedOptimizer() 