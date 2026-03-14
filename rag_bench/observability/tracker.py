"""In-memory request tracker for frontend metrics summary.

Collects per-query stats in ring buffers for exact percentile computation,
tracks unique users by IP, and reads live hardware stats via psutil/pynvml.
"""

from __future__ import annotations

import os
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

import psutil

try:
    from pynvml import (
        NVML_TEMPERATURE_GPU,
        NVMLError,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName,
        nvmlDeviceGetTemperature,
        nvmlDeviceGetUtilizationRates,
        nvmlInit,
        nvmlShutdown,
    )

    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False


def _percentile(sorted_data: list[float], p: float) -> float:
    """Compute the p-th percentile from pre-sorted data."""
    if not sorted_data:
        return 0.0
    idx = int(len(sorted_data) * p)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def _get_gpu_stats() -> list[dict[str, Any]]:
    """Read GPU utilization, VRAM, and temperature for all NVIDIA GPUs.

    Temporarily clears CUDA_VISIBLE_DEVICES so NVML reports *all* physical
    GPUs, not just the ones visible to the CUDA runtime.

    Returns an empty list if no NVIDIA GPU or pynvml is unavailable.
    """
    if not _HAS_PYNVML:
        return []
    try:
        # NVML respects CUDA_VISIBLE_DEVICES — temporarily unset it so we
        # see every physical GPU for monitoring purposes.
        cuda_env = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            nvmlInit()
            gpu_count = nvmlDeviceGetCount()
            gpus = []
            for i in range(gpu_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                name = nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                try:
                    temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                except NVMLError:
                    temp = None
                gpus.append(
                    {
                        "name": name,
                        "gpu_util_percent": util.gpu,
                        "vram_used_gb": round(mem.used / (1024**3), 1),
                        "vram_total_gb": round(mem.total / (1024**3), 1),
                        "temperature_c": temp,
                    }
                )
            nvmlShutdown()
        finally:
            if cuda_env is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_env
        return gpus
    except Exception:
        return []


class RequestTracker:
    """Lightweight in-memory stats collector for the /api/metrics/summary endpoint.

    Thread-safe for single-writer (the FastAPI event loop) usage pattern.
    """

    _BUFFER_SIZE = 1000
    _RECENT_SIZE = 50
    _TIMESERIES_MAX_POINTS = 500

    def __init__(self) -> None:
        self.start_time = time.monotonic()
        self.total_queries = 0
        self.status_counts: dict[str, int] = {"success": 0, "deflected": 0, "error": 0}
        self.unique_ips: set[str] = set()

        self._latencies: deque[float] = deque(maxlen=self._BUFFER_SIZE)
        self._retrieval_times: deque[float] = deque(maxlen=self._BUFFER_SIZE)
        self._generation_times: deque[float] = deque(maxlen=self._BUFFER_SIZE)
        self._reranking_times: deque[float] = deque(maxlen=self._BUFFER_SIZE)
        self._citation_coverages: deque[float] = deque(maxlen=self._BUFFER_SIZE)
        self._top_scores: deque[float] = deque(maxlen=self._BUFFER_SIZE)

        self._recent_queries: deque[dict[str, Any]] = deque(maxlen=self._RECENT_SIZE)

        # Per-query rolling percentile snapshots for the time-series chart
        self._latency_history: deque[dict[str, Any]] = deque(maxlen=self._TIMESERIES_MAX_POINTS)

    def record_query(
        self,
        *,
        latency_ms: float,
        status: str,
        question_preview: str,
        retrieval_ms: float = 0.0,
        generation_ms: float = 0.0,
        reranking_ms: float = 0.0,
        citation_coverage: float = 0.0,
        top_score: float = 0.0,
        client_ip: str = "",
    ) -> None:
        """Record a single query's metrics."""
        self.total_queries += 1
        self.status_counts[status] = self.status_counts.get(status, 0) + 1

        if client_ip:
            self.unique_ips.add(client_ip)

        self._latencies.append(latency_ms)
        if retrieval_ms > 0:
            self._retrieval_times.append(retrieval_ms)
        if generation_ms > 0:
            self._generation_times.append(generation_ms)
        if reranking_ms > 0:
            self._reranking_times.append(reranking_ms)
        if citation_coverage > 0:
            self._citation_coverages.append(citation_coverage)
        if top_score > 0:
            self._top_scores.append(top_score)

        now = datetime.now(timezone.utc).isoformat()

        self._recent_queries.append(
            {
                "timestamp": now,
                "question": question_preview[:80],
                "latency_ms": round(latency_ms, 1),
                "status": status,
            }
        )

        # Rolling percentile snapshot for the time-series chart
        s = sorted(self._latencies)
        self._latency_history.append(
            {
                "t": now,
                "p50": round(_percentile(s, 0.50), 1),
                "p90": round(_percentile(s, 0.90), 1),
                "p99": round(_percentile(s, 0.99), 1),
                "retrieval_ms": round(retrieval_ms, 1),
                "generation_ms": round(generation_ms, 1),
                "reranking_ms": round(reranking_ms, 1),
                "n": len(s),
            }
        )

    def summary(self) -> dict[str, Any]:
        """Return the full metrics summary for the frontend."""
        sorted_lat = sorted(self._latencies)

        def _avg(d: deque[float]) -> float:
            return round(sum(d) / len(d), 1) if d else 0.0

        uptime = time.monotonic() - self.start_time
        ram = psutil.virtual_memory()

        return {
            "total_queries": self.total_queries,
            "unique_users": len(self.unique_ips),
            "uptime_seconds": round(uptime),
            "queries_by_status": dict(self.status_counts),
            "latency": {
                "avg_ms": _avg(self._latencies),
                "p50_ms": round(_percentile(sorted_lat, 0.50), 1),
                "p90_ms": round(_percentile(sorted_lat, 0.90), 1),
                "p99_ms": round(_percentile(sorted_lat, 0.99), 1),
            },
            "pipeline": {
                "avg_retrieval_ms": _avg(self._retrieval_times),
                "avg_generation_ms": _avg(self._generation_times),
                "avg_reranking_ms": _avg(self._reranking_times),
            },
            "quality": {
                "avg_citation_coverage": round(_avg(self._citation_coverages) if self._citation_coverages else 0.0, 3),
                "avg_top_score": round(_avg(self._top_scores) if self._top_scores else 0.0, 3),
            },
            "hardware": {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "ram_used_gb": round(ram.used / (1024**3), 1),
                "ram_total_gb": round(ram.total / (1024**3), 1),
                "ram_percent": ram.percent,
                "gpus": _get_gpu_stats(),
            },
            "recent_queries": list(reversed(self._recent_queries)),
            "latency_history": list(self._latency_history),
        }
