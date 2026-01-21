"""
GPU Monitor Module
Monitors NVIDIA GPU utilization and memory usage.
"""

import threading
import time
from typing import Dict, Optional, Tuple

from .logger import get_logger

logger = get_logger(__name__)

# Try to import pynvml for GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available. GPU monitoring disabled.")


class GPUMonitor:
    """
    Monitors NVIDIA GPU statistics.
    
    Tracks:
    - GPU utilization percentage
    - Memory usage (used/total)
    - Temperature
    - Power consumption
    
    Attributes:
        device_index: GPU device index to monitor
    """
    
    def __init__(self, device_index: int = 0):
        """
        Initialize GPU monitor.
        
        Args:
            device_index: Index of GPU to monitor (0 for first GPU)
        """
        self.device_index = device_index
        self._handle = None
        self._initialized = False
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stats_history: list = []
        self._max_history = 1000
        
        if PYNVML_AVAILABLE:
            self._initialize()
    
    def _initialize(self) -> None:
        """Initialize NVML library and get device handle."""
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            if device_count == 0:
                logger.warning("No NVIDIA GPUs found")
                return
            
            if self.device_index >= device_count:
                logger.warning(f"GPU index {self.device_index} not available. Using GPU 0.")
                self.device_index = 0
            
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            name = pynvml.nvmlDeviceGetName(self._handle)
            
            # Handle both bytes and string return types
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            logger.info(f"GPU Monitor initialized: {name}")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU monitor: {e}")
            self._initialized = False
    
    def is_available(self) -> bool:
        """Check if GPU monitoring is available."""
        return PYNVML_AVAILABLE and self._initialized
    
    def get_utilization(self) -> float:
        """
        Get current GPU utilization.
        
        Returns:
            GPU utilization percentage (0-100), or -1 if unavailable
        """
        if not self.is_available():
            return -1.0
        
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            return float(util.gpu)
        except Exception as e:
            logger.debug(f"Failed to get GPU utilization: {e}")
            return -1.0
    
    def get_memory(self) -> Tuple[float, float, float]:
        """
        Get GPU memory statistics.
        
        Returns:
            Tuple of (used_mb, total_mb, percentage)
        """
        if not self.is_available():
            return (-1.0, -1.0, -1.0)
        
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            used_mb = info.used / (1024 ** 2)
            total_mb = info.total / (1024 ** 2)
            percentage = (info.used / info.total) * 100
            return (used_mb, total_mb, percentage)
        except Exception as e:
            logger.debug(f"Failed to get GPU memory: {e}")
            return (-1.0, -1.0, -1.0)
    
    def get_temperature(self) -> float:
        """
        Get GPU temperature.
        
        Returns:
            Temperature in Celsius, or -1 if unavailable
        """
        if not self.is_available():
            return -1.0
        
        try:
            temp = pynvml.nvmlDeviceGetTemperature(
                self._handle,
                pynvml.NVML_TEMPERATURE_GPU
            )
            return float(temp)
        except Exception as e:
            logger.debug(f"Failed to get GPU temperature: {e}")
            return -1.0
    
    def get_power(self) -> Tuple[float, float]:
        """
        Get GPU power consumption.
        
        Returns:
            Tuple of (current_watts, limit_watts)
        """
        if not self.is_available():
            return (-1.0, -1.0)
        
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0  # mW to W
            limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(self._handle) / 1000.0
            return (power, limit)
        except Exception as e:
            logger.debug(f"Failed to get GPU power: {e}")
            return (-1.0, -1.0)
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get all GPU statistics.
        
        Returns:
            Dictionary with all GPU stats
        """
        used_mb, total_mb, mem_percent = self.get_memory()
        power, power_limit = self.get_power()
        
        return {
            'utilization': self.get_utilization(),
            'memory_used_mb': used_mb,
            'memory_total_mb': total_mb,
            'memory_percent': mem_percent,
            'temperature': self.get_temperature(),
            'power_watts': power,
            'power_limit_watts': power_limit,
        }
    
    def check_memory_warning(self, threshold_percent: float = 90.0) -> bool:
        """
        Check if GPU memory usage is above threshold.
        
        Args:
            threshold_percent: Warning threshold percentage
            
        Returns:
            True if memory usage exceeds threshold
        """
        _, _, mem_percent = self.get_memory()
        if mem_percent > threshold_percent:
            logger.warning(f"GPU memory usage high: {mem_percent:.1f}%")
            return True
        return False
    
    def start_monitoring(self, interval: float = 5.0) -> None:
        """
        Start background monitoring thread.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info(f"GPU monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        logger.info("GPU monitoring stopped")
    
    def _monitor_loop(self, interval: float) -> None:
        """Background monitoring loop."""
        while self._monitoring:
            stats = self.get_stats()
            stats['timestamp'] = time.time()
            
            self._stats_history.append(stats)
            
            # Trim history
            if len(self._stats_history) > self._max_history:
                self._stats_history = self._stats_history[-self._max_history:]
            
            # Check for warnings
            self.check_memory_warning()
            
            time.sleep(interval)
    
    def get_history(self) -> list:
        """Get statistics history."""
        return self._stats_history.copy()
    
    def get_average_stats(self, last_n: int = 100) -> Dict[str, float]:
        """
        Get average statistics over last N samples.
        
        Args:
            last_n: Number of recent samples to average
            
        Returns:
            Dictionary with average stats
        """
        if not self._stats_history:
            return self.get_stats()
        
        recent = self._stats_history[-last_n:]
        
        avg_stats = {}
        keys = ['utilization', 'memory_used_mb', 'memory_percent', 'temperature', 'power_watts']
        
        for key in keys:
            values = [s[key] for s in recent if s.get(key, -1) >= 0]
            avg_stats[key] = sum(values) / len(values) if values else -1.0
        
        return avg_stats
    
    def shutdown(self) -> None:
        """Shutdown GPU monitor and cleanup."""
        self.stop_monitoring()
        
        if PYNVML_AVAILABLE and self._initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        
        self._initialized = False
    
    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()


# Convenience function
def get_gpu_memory_usage() -> Tuple[float, float]:
    """
    Quick function to get GPU memory usage.
    
    Returns:
        Tuple of (used_mb, total_mb)
    """
    monitor = GPUMonitor()
    used, total, _ = monitor.get_memory()
    return (used, total)
