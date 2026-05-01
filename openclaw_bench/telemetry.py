from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GpuSample:
    index: int
    name: str
    memory_total_mb: float
    memory_used_mb: float
    utilization_pct: float

    def to_row(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "memory_total_mb": self.memory_total_mb,
            "memory_used_mb": self.memory_used_mb,
            "utilization_pct": self.utilization_pct,
        }


@dataclass
class GpuTelemetry:
    available: bool
    samples: list[GpuSample] = field(default_factory=list)
    error: str = ""

    @property
    def peak_vram_mb(self) -> float | None:
        if not self.samples:
            return None
        return max(sample.memory_used_mb for sample in self.samples)

    @property
    def max_gpu_utilization_pct(self) -> float | None:
        if not self.samples:
            return None
        return max(sample.utilization_pct for sample in self.samples)

    def to_row(self) -> dict:
        latest_by_index = {sample.index: sample for sample in self.samples}
        return {
            "available": self.available,
            "devices": [sample.to_row() for sample in sorted(latest_by_index.values(), key=lambda item: item.index)],
            "peak_vram_mb": self.peak_vram_mb,
            "max_gpu_utilization_pct": self.max_gpu_utilization_pct,
            "error": self.error,
        }


class GpuTelemetrySampler:
    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[GpuSample] = []
        self._error = ""
        self._available = shutil.which("nvidia-smi") is not None

    def __enter__(self) -> "GpuTelemetrySampler":
        if not self._available:
            self._error = "nvidia-smi not found"
            return self
        self._sample_once()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 2))
        if self._available:
            self._sample_once()

    def result(self) -> GpuTelemetry:
        return GpuTelemetry(available=self._available, samples=list(self._samples), error=self._error)

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample_once()

    def _sample_once(self) -> None:
        sample = sample_nvidia_smi()
        if sample.available:
            self._samples.extend(sample.samples)
        elif sample.error:
            self._error = sample.error


def sample_nvidia_smi() -> GpuTelemetry:
    if shutil.which("nvidia-smi") is None:
        return GpuTelemetry(available=False, error="nvidia-smi not found")
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return GpuTelemetry(available=False, error=str(exc))
    if proc.returncode != 0:
        return GpuTelemetry(available=False, error=(proc.stderr or proc.stdout).strip())
    try:
        return GpuTelemetry(available=True, samples=parse_nvidia_smi(proc.stdout))
    except ValueError as exc:
        return GpuTelemetry(available=False, error=str(exc))


def parse_nvidia_smi(output: str) -> list[GpuSample]:
    samples = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise ValueError(f"unexpected nvidia-smi row: {line}")
        samples.append(
            GpuSample(
                index=int(parts[0]),
                name=parts[1],
                memory_total_mb=float(parts[2]),
                memory_used_mb=float(parts[3]),
                utilization_pct=float(parts[4]),
            )
        )
    return samples


def apply_gpu_telemetry(target, telemetry: GpuTelemetry) -> None:
    target.peak_vram_mb = telemetry.peak_vram_mb
    if hasattr(target, "gpu_utilization_pct"):
        target.gpu_utilization_pct = telemetry.max_gpu_utilization_pct


def merge_gpu_telemetry(items: list[GpuTelemetry]) -> GpuTelemetry:
    samples: list[GpuSample] = []
    errors = []
    available = False
    for item in items:
        available = available or item.available
        samples.extend(item.samples)
        if item.error:
            errors.append(item.error)
    return GpuTelemetry(available=available, samples=samples, error="; ".join(errors))
