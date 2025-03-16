import pynvml
import logging
from typing import Dict, Optional
from dataclasses import dataclass
import time

@dataclass
class GPUStats:
    memory_total: int
    memory_used: int
    utilization: int
    temperature: int
    power_usage: float
    compute_mode: str
    gpu_name: str
    uuid: str

class GPUMonitor:
    def __init__(self):
        self.initialized = False
        self._init_nvml()
        
    def _init_nvml(self):
        """Initialize NVIDIA Management Library"""
        try:
            pynvml.nvmlInit()
            self.initialized = True
            self.device_count = pynvml.nvmlDeviceGetCount()
            logging.info(f"NVML initialized. Found {self.device_count} GPU devices.")
        except Exception as e:
            logging.error(f"Failed to initialize NVML: {str(e)}")
            self.initialized = False

    def get_gpu_info(self) -> Dict:
        """Get detailed GPU information"""
        if not self.initialized:
            return {}
            
        try:
            device = pynvml.nvmlDeviceGetHandleByIndex(0)  # Using first GPU
            
            # Get device info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(device)
            compute_mode = pynvml.nvmlDeviceGetComputeMode(device)
            gpu_name = pynvml.nvmlDeviceGetName(device)
            uuid = pynvml.nvmlDeviceGetUUID(device)
            
            # Get compute capability
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(device)
            
            return {
                "gpu_model": gpu_name.decode() if isinstance(gpu_name, bytes) else gpu_name,
                "memory_total": memory_info.total,
                "compute_capability_major": major,
                "compute_capability_minor": minor,
                "uuid": uuid.decode() if isinstance(uuid, bytes) else uuid,
                "compute_mode": self._get_compute_mode_string(compute_mode),
                "supported_precisions": self._get_supported_precisions(major, minor)
            }
        except Exception as e:
            logging.error(f"Error getting GPU info: {str(e)}")
            return {}

    def get_current_stats(self) -> Optional[GPUStats]:
        """Get current GPU statistics"""
        if not self.initialized:
            return None
            
        try:
            device = pynvml.nvmlDeviceGetHandleByIndex(0)  # Using first GPU
            
            # Get current metrics
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(device)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(device)
            temperature = pynvml.nvmlDeviceGetTemperature(device, pynvml.NVML_TEMPERATURE_GPU)
            power_usage = pynvml.nvmlDeviceGetPowerUsage(device) / 1000.0  # Convert to watts
            
            return GPUStats(
                memory_total=memory_info.total,
                memory_used=memory_info.used,
                utilization=utilization.gpu,
                temperature=temperature,
                power_usage=power_usage,
                compute_mode=self._get_compute_mode_string(pynvml.nvmlDeviceGetComputeMode(device)),
                gpu_name=pynvml.nvmlDeviceGetName(device).decode(),
                uuid=pynvml.nvmlDeviceGetUUID(device).decode()
            )
        except Exception as e:
            logging.error(f"Error getting GPU stats: {str(e)}")
            return None

    def _get_compute_mode_string(self, compute_mode) -> str:
        """Convert compute mode enum to string"""
        modes = {
            pynvml.NVML_COMPUTEMODE_DEFAULT: "Default",
            pynvml.NVML_COMPUTEMODE_EXCLUSIVE_THREAD: "Exclusive Thread",
            pynvml.NVML_COMPUTEMODE_PROHIBITED: "Prohibited",
            pynvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS: "Exclusive Process"
        }
        return modes.get(compute_mode, "Unknown")

    def _get_supported_precisions(self, major: int, minor: int) -> list:
        """Determine supported precision formats based on compute capability"""
        supported = ["fp32"]
        
        # FP16 support
        if major >= 6:
            supported.append("fp16")
            
        # INT8 support
        if major >= 6:
            supported.append("int8")
            
        # TF32 support
        if major >= 8:
            supported.append("tf32")
            
        # BF16 support
        if major >= 8:
            supported.append("bf16")
            
        return supported

    def cleanup(self):
        """Cleanup NVML"""
        if self.initialized:
            try:
                pynvml.nvmlShutdown()
                self.initialized = False
            except Exception as e:
                logging.error(f"Error shutting down NVML: {str(e)}")

    def __del__(self):
        """Ensure cleanup on deletion"""
        self.cleanup()