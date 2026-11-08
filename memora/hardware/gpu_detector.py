"""Hardware discovery and device capability detection for CPU, CUDA, and MPS."""

from __future__ import annotations

import os
import platform
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import psutil
import torch

from memora.utils.common import format_bytes


@dataclass
class DeviceInfo:
    """Detailed hardware information for a single compute device."""

    device_type: str  # 'cuda', 'mps', or 'cpu'
    device_index: int
    name: str
    total_memory: int  # in bytes
    free_memory: Optional[int] = None  # in bytes
    compute_capability: Optional[str] = None  # e.g., '8.6'
    is_available: bool = True
    supports_fp16: bool = False
    supports_bf16: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary of the device."""
        mem_str = format_bytes(self.total_memory)
        cc_str = f" (Compute Capability: {self.compute_capability})" if self.compute_capability else ""
        return f"[{self.device_type.upper()}:{self.device_index}] {self.name} | Total Memory: {mem_str}{cc_str}"


@dataclass
class HardwareReport:
    """Comprehensive system-wide hardware and environment report."""

    platform_system: str
    platform_release: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    device_count: int
    default_device: DeviceInfo
    devices: List[DeviceInfo] = field(default_factory=list)
    system_ram_total: int = 0
    system_ram_available: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "platform_system": self.platform_system,
            "platform_release": self.platform_release,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "device_count": self.device_count,
            "system_ram_total": self.system_ram_total,
            "system_ram_available": self.system_ram_available,
            "system_ram_total_str": format_bytes(self.system_ram_total),
            "system_ram_available_str": format_bytes(self.system_ram_available),
            "default_device": asdict(self.default_device),
            "devices": [asdict(d) for d in self.devices],
        }


class GPUDetector:
    """Detects available GPU/CPU hardware, device capabilities, and memory properties."""

    @classmethod
    def is_cuda_available(cls) -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()

    @classmethod
    def is_mps_available(cls) -> bool:
        """Check if Apple Silicon MPS is available."""
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    @classmethod
    def get_nvml_memory(cls, device_index: int = 0) -> Optional[Dict[str, int]]:
        """Query device-level memory via NVML if nvidia-ml-py is installed."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            return {
                "total": int(mem_info.total),
                "free": int(mem_info.free),
                "used": int(mem_info.used),
            }
        except Exception:
            return None

    @classmethod
    def get_device_info(cls, device: Optional[torch.device] = None) -> DeviceInfo:
        """Detect and return information for the specified device."""
        if device is None:
            if cls.is_cuda_available():
                device = torch.device("cuda:0")
            elif cls.is_mps_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        if device.type == "cuda":
            idx = device.index or 0
            name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            total_mem = props.total_memory
            major, minor = props.major, props.minor
            compute_cap = f"{major}.{minor}"

            # PyTorch CUDA memory API
            try:
                free_mem, _ = torch.cuda.mem_get_info(idx)
            except Exception:
                free_mem = None

            # Check bf16 / fp16 capabilities:
            # BF16 is natively supported on Ampere (Compute Capability >= 8.0)
            supports_bf16 = major >= 8 or (hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
            supports_fp16 = major >= 6  # Pascal and newer have native FP16 support

            nvml_mem = cls.get_nvml_memory(idx)

            return DeviceInfo(
                device_type="cuda",
                device_index=idx,
                name=name,
                total_memory=total_mem,
                free_memory=free_mem,
                compute_capability=compute_cap,
                is_available=True,
                supports_fp16=supports_fp16,
                supports_bf16=supports_bf16,
                details={
                    "multi_processor_count": getattr(props, "multi_processor_count", None),
                    "nvml_device_memory": nvml_mem,
                },
            )

        if device.type == "mps":
            # Apple Silicon MPS
            # MPS uses unified memory from host RAM
            vm = psutil.virtual_memory()
            return DeviceInfo(
                device_type="mps",
                device_index=0,
                name="Apple Silicon MPS",
                total_memory=vm.total,
                free_memory=vm.available,
                compute_capability=None,
                is_available=True,
                supports_fp16=True,
                supports_bf16=False,
                details={"unified_memory": True},
            )

        # Fallback to CPU
        vm = psutil.virtual_memory()
        cpu_count = os.cpu_count() or 1
        return DeviceInfo(
            device_type="cpu",
            device_index=0,
            name=f"CPU ({platform.processor() or platform.machine()}, {cpu_count} cores)",
            total_memory=vm.total,
            free_memory=vm.available,
            compute_capability=None,
            is_available=True,
            supports_fp16=False,
            supports_bf16=hasattr(torch, "bfloat16"),
            details={"cpu_count": cpu_count},
        )

    @classmethod
    def get_hardware_report(cls) -> HardwareReport:
        """Scan system and produce comprehensive hardware report."""
        vm = psutil.virtual_memory()
        cuda_avail = cls.is_cuda_available()
        cuda_version = torch.version.cuda if cuda_avail else None

        devices: List[DeviceInfo] = []
        if cuda_avail:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                devices.append(cls.get_device_info(torch.device(f"cuda:{i}")))
            default_device = devices[0]
        elif cls.is_mps_available():
            mps_dev = cls.get_device_info(torch.device("mps"))
            devices.append(mps_dev)
            default_device = mps_dev
            device_count = 1
        else:
            cpu_dev = cls.get_device_info(torch.device("cpu"))
            devices.append(cpu_dev)
            default_device = cpu_dev
            device_count = 1

        return HardwareReport(
            platform_system=platform.system(),
            platform_release=platform.release(),
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            cuda_available=cuda_avail,
            cuda_version=cuda_version,
            device_count=device_count,
            default_device=default_device,
            devices=devices,
            system_ram_total=vm.total,
            system_ram_available=vm.available,
        )

