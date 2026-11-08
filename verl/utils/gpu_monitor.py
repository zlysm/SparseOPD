# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Background NVIDIA GPU sampling for trainer metrics."""

import csv
import io
import logging
import math
import os
import subprocess
import threading

logger = logging.getLogger(__name__)


class GpuStatsMonitor:
    """Sample nvidia-smi in the background and expose per-GPU step averages."""

    QUERY_FIELDS = (
        "index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
    )

    def __init__(self, sample_interval_s: float = 2.0):
        if sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be positive")
        self.sample_interval_s = sample_interval_s
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        self.visible_devices = {device.strip() for device in visible_devices.split(",") if device.strip()}
        self._lock = threading.Lock()
        self._samples: list[dict[str, float]] = []
        self._stop_event = threading.Event()
        self._warned = False
        self._sample_once()
        self._thread = threading.Thread(target=self._run, name="gpu-stats-monitor", daemon=True)
        self._thread.start()

    @classmethod
    def parse_csv(cls, output: str, visible_devices: set[str] | None = None) -> dict[str, float]:
        """Parse nvidia-smi CSV output into TensorBoard-ready metrics."""
        result = {}
        reader = csv.reader(io.StringIO(output), skipinitialspace=True)
        for row in reader:
            if len(row) != 7:
                continue
            index, uuid, gpu_util, memory_util, memory_used, memory_total, power = [x.strip() for x in row]
            try:
                index = str(int(index))
            except ValueError:
                continue
            if visible_devices and index not in visible_devices and uuid not in visible_devices:
                continue
            prefix = f"gpu/{index}"

            values = {
                "utilization_gpu_pct": gpu_util,
                "utilization_memory_pct": memory_util,
                "memory_used_gb": memory_used,
                "memory_total_gb": memory_total,
                "power_w": power,
            }
            for name, raw in values.items():
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if not math.isfinite(value):
                    continue
                if name.startswith("memory_"):
                    value /= 1024.0
                result[f"{prefix}/{name}"] = value
        return result

    def _sample_once(self) -> None:
        try:
            completed = subprocess.run(
                ["nvidia-smi", f"--query-gpu={self.QUERY_FIELDS}", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            sample = self.parse_csv(completed.stdout, self.visible_devices or None)
            if not sample:
                raise RuntimeError("nvidia-smi returned no GPU metrics")
        except Exception as exc:
            if not self._warned:
                logger.warning("GPU metrics unavailable; nvidia-smi sampling failed: %s", exc)
                self._warned = True
            return
        with self._lock:
            self._samples.append(sample)

    def _run(self) -> None:
        while not self._stop_event.wait(self.sample_interval_s):
            self._sample_once()

    def collect(self) -> dict[str, float]:
        """Return mean metrics since the previous call, including per-GPU and fleet means."""
        with self._lock:
            samples, self._samples = self._samples, []
        if not samples:
            return {}

        totals: dict[str, list[float]] = {}
        for sample in samples:
            for key, value in sample.items():
                totals.setdefault(key, []).append(value)
        metrics = {key: sum(values) / len(values) for key, values in totals.items()}

        gpu_keys = sorted(
            {
                key.split("/")[1]
                for key in metrics
                if key.startswith("gpu/") and key.split("/")[1].isdigit()
            },
            key=int,
        )
        for suffix in (
            "utilization_gpu_pct",
            "utilization_memory_pct",
            "memory_used_gb",
            "memory_total_gb",
            "power_w",
        ):
            values = [metrics[f"gpu/{gpu}/{suffix}"] for gpu in gpu_keys if f"gpu/{gpu}/{suffix}" in metrics]
            if values:
                metrics[f"gpu/mean/{suffix}"] = sum(values) / len(values)
        metrics["gpu/sample_count"] = float(len(samples))
        return metrics

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self.sample_interval_s + 1.0))

