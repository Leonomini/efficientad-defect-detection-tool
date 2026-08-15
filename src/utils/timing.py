import time
import torch

from contextlib import contextmanager
from numbers import Number
from src.utils.cli_utils import print_kv


def _sync_if_cuda(device):
    if device is None:
        return
    device_str = str(device)
    if "cuda" in device_str:
        torch.cuda.synchronize()


@contextmanager
def timed_block(name: str, timings: dict, device=None):
    _sync_if_cuda(device)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _sync_if_cuda(device)
        t1 = time.perf_counter()
        timings[name] = timings.get(name, 0.0) + (t1 - t0)


class TimingCollector:
    def __init__(self):
        self.per_sample = []  # list[dict[str,float]]

    def add(self, sample_timing: dict):
        self.per_sample.append(sample_timing)

    def summary(self):
        # returns dict of mean times (seconds) per key
        if not self.per_sample:
            return {}

        keys = set()
        for d in self.per_sample:
            keys |= set(d.keys())

        out = {}
        for k in sorted(keys):
            vals = [d.get(k, 0.0) for d in self.per_sample]
            out[k] = sum(vals) / len(vals)

        out["processed_samples"] = len(self.per_sample)
        return out


def compute_pipeline_timings(
    total_seconds: float,
    num_samples: int,
) -> dict:
    if total_seconds <= 0 or num_samples <= 0:
        return {
            "run_total_samples": num_samples,
            "run_total_seconds": total_seconds,
            "avg_ms_per_sample": 0.0,
            "avg_fps": 0.0,
        }

    return {
        "run_total_samples": num_samples,
        "run_total_seconds": total_seconds,
        "avg_ms_per_sample": (total_seconds / num_samples) * 1000.0,
        "avg_fps": num_samples / total_seconds,
    }


def print_pipeline_timings(summary: dict):
    if not summary:
        return

    print("\n[Pipeline Performance Summary]")
    print_kv("Processed samples", summary["run_total_samples"])
    print_kv("Total runtime", f"{summary['run_total_seconds']:.2f} s")
    print_kv("Latency per sample", f"{summary['avg_ms_per_sample']:.2f} ms")
    print_kv("Throughput", f"{summary['avg_fps']:.2f} img/s")
    print()


def print_sample_timings(summary: dict):
    if not summary:
        return

    print("\n[Per-Sample Inference Breakdown]")
    if "processed_samples" in summary:
        print_kv("processed_samples", summary["processed_samples"])

    for k, v in summary.items():
        if k == "processed_samples":
            continue

        if isinstance(v, Number):
            print_kv(k, f"{float(v) * 1000:.2f} ms")
        else:
            print_kv(k, v)

