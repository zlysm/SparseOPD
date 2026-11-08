import threading

from verl.utils.gpu_monitor import GpuStatsMonitor


def test_parse_csv_creates_per_gpu_memory_and_utilization_metrics():
    output = """0, GPU-a, 37, 21, 16384, 81920, 250.5
1, GPU-b, 63, 42, 32768, 81920, 300.0
"""

    metrics = GpuStatsMonitor.parse_csv(output)

    assert metrics["gpu/0/utilization_gpu_pct"] == 37
    assert metrics["gpu/0/utilization_memory_pct"] == 21
    assert metrics["gpu/0/memory_used_gb"] == 16
    assert metrics["gpu/1/memory_used_gb"] == 32
    assert metrics["gpu/1/memory_total_gb"] == 80
    assert metrics["gpu/1/power_w"] == 300


def test_parse_csv_skips_unavailable_values():
    metrics = GpuStatsMonitor.parse_csv("0, GPU-a, N/A, N/A, 8192, 81920, N/A\n")

    assert "gpu/0/utilization_gpu_pct" not in metrics
    assert metrics["gpu/0/memory_used_gb"] == 8
    assert "gpu/0/power_w" not in metrics


def test_collect_averages_samples_per_gpu_and_across_gpus():
    monitor = GpuStatsMonitor.__new__(GpuStatsMonitor)
    monitor._lock = threading.Lock()
    monitor._samples = [
        {"gpu/0/utilization_gpu_pct": 20, "gpu/1/utilization_gpu_pct": 60},
        {"gpu/0/utilization_gpu_pct": 40, "gpu/1/utilization_gpu_pct": 80},
    ]

    metrics = monitor.collect()

    assert metrics["gpu/0/utilization_gpu_pct"] == 30
    assert metrics["gpu/1/utilization_gpu_pct"] == 70
    assert metrics["gpu/mean/utilization_gpu_pct"] == 50
    assert metrics["gpu/sample_count"] == 2
    assert monitor.collect() == {}

