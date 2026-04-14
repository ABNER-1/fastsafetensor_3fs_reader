#!/usr/bin/env python3
"""Diagnose GPU resource contention between producer and consumer."""

import os
import subprocess
import sys
import threading
import time
from typing import Dict, List

import torch


class GPUMonitor:

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.monitoring = False
        self.samples: List[Dict] = []
        self.monitor_thread = None

    def start(self):

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)

    def _monitor_loop(self):

        while self.monitoring:
            try:

                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free",
                        "--format=csv,noheader,nounits",
                        f"--id={self.device_id}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1,
                )

                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    if len(parts) >= 5:
                        self.samples.append(
                            {
                                "timestamp": time.time(),
                                "gpu_util": float(parts[1]),
                                "mem_util": float(parts[2]),
                                "mem_used_mb": float(parts[3]),
                                "mem_free_mb": float(parts[4]),
                            }
                        )

                time.sleep(0.1)  # 100 ms sampling interval

            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(0.5)

    def get_statistics(self) -> Dict:

        if not self.samples:
            return {}

        gpu_utils = [s["gpu_util"] for s in self.samples]
        mem_utils = [s["mem_util"] for s in self.samples]

        return {
            "sample_count": len(self.samples),
            "avg_gpu_util": sum(gpu_utils) / len(gpu_utils),
            "max_gpu_util": max(gpu_utils),
            "avg_mem_util": sum(mem_utils) / len(mem_utils),
            "max_mem_util": max(mem_utils),
            "peak_mem_used_mb": max(s["mem_used_mb"] for s in self.samples),
        }


class CUDAProfiler:

    def __init__(self):
        self.events: List[Dict] = []
        self.current_phase = None
        self.phase_start = None

    def start_phase(self, phase_name: str):

        if self.current_phase:
            self.end_phase()

        self.current_phase = phase_name
        self.phase_start = time.time()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def end_phase(self):

        if not self.current_phase:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        duration = time.time() - self.phase_start

        self.events.append(
            {
                "phase": self.current_phase,
                "duration": duration,
                "timestamp": self.phase_start,
            }
        )

        self.current_phase = None
        self.phase_start = None

    def get_summary(self) -> Dict:

        if not self.events:
            return {}

        phase_times = {}
        for event in self.events:
            phase = event["phase"]
            if phase not in phase_times:
                phase_times[phase] = []
            phase_times[phase].append(event["duration"])

        summary = {}
        for phase, times in phase_times.items():
            summary[phase] = {
                "count": len(times),
                "total": sum(times),
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
            }

        return summary


def test_with_profiling(test_files: List[str], queue_size: int):

    from fastsafetensors import ParallelLoader

    print(f"\n{'='*80}")
    print(f"Testing queue_size={queue_size} (with GPU monitoring)")
    print(f"{'='*80}\n")

    gpu_monitor = GPUMonitor(device_id=0)
    gpu_monitor.start()

    cuda_profiler = CUDAProfiler()

    try:

        os.environ["FASTSAFETENSORS_DEBUG"] = "true"

        start_time = time.time()

        iterator = ParallelLoader(
            pg=None,
            hf_weights_files=test_files,
            max_concurrent_producers=1,
            queue_size=queue_size,
            use_tqdm_on_load=True,
            device="cuda:0",
            nogds=False,
            debug_log=True,
        )

        tensor_count = 0
        get_tensor_start = time.time()

        for key, tensor in iterator.iterate_weights():

            cuda_profiler.start_phase(f"get_tensor_{tensor_count}")

            cuda_profiler.end_phase()
            tensor_count += 1

        get_tensor_end = time.time()
        iterator.close()

        end_time = time.time()

        gpu_monitor.stop()

        print(f"\n{'='*80}")
        print(f"Results (queue_size={queue_size})")
        print(f"{'='*80}")
        print(f"Total time: {end_time - start_time:.2f}s")
        print(f"get_tensor total: {get_tensor_end - get_tensor_start:.2f}s")
        print(f"Tensor count: {tensor_count}")

        gpu_stats = gpu_monitor.get_statistics()
        if gpu_stats:
            print(f"\nGPU usage stats:")
            print(f"  Avg GPU util: {gpu_stats['avg_gpu_util']:.1f}%")
            print(f"  Peak GPU util: {gpu_stats['max_gpu_util']:.1f}%")
            print(f"  Avg mem util: {gpu_stats['avg_mem_util']:.1f}%")
            print(f"  Peak mem used: {gpu_stats['peak_mem_used_mb']:.0f} MB")

        cuda_summary = cuda_profiler.get_summary()
        if cuda_summary:
            print(f"\nCUDA op stats:")
            for phase, stats in cuda_summary.items():
                print(f"  {phase}:")
                print(f"    Count: {stats['count']}")
                print(f"    Total: {stats['total']:.3f}s")
                print(f"    Avg: {stats['avg']*1000:.2f} ms")
                print(
                    f"    Range: {stats['min']*1000:.2f} - {stats['max']*1000:.2f} ms"
                )

        print(f"{'='*80}\n")

        return {
            "total_time": end_time - start_time,
            "get_tensor_time": get_tensor_end - get_tensor_start,
            "tensor_count": tensor_count,
            "gpu_stats": gpu_stats,
            "cuda_summary": cuda_summary,
        }

    except Exception as e:
        gpu_monitor.stop()
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def compare_gpu_usage(results_0: Dict, results_neg1: Dict):

    print(f"\n{'='*80}")
    print("GPU resource contention analysis")
    print(f"{'='*80}\n")

    gpu_0 = results_0.get("gpu_stats", {})
    gpu_neg1 = results_neg1.get("gpu_stats", {})

    if gpu_0 and gpu_neg1:
        print(f"{'Metric':<30} {'queue_size=0':>15} {'queue_size=-1':>15} {'Diff':>15}")
        print(f"{'-'*80}")
        print(
            f"{'Avg GPU util (%)' :<30} {gpu_0.get('avg_gpu_util', 0):>15.1f} "
            f"{gpu_neg1.get('avg_gpu_util', 0):>15.1f} "
            f"{gpu_0.get('avg_gpu_util', 0) - gpu_neg1.get('avg_gpu_util', 0):>14.1f}"
        )
        print(
            f"{'Peak GPU util (%)' :<30} {gpu_0.get('max_gpu_util', 0):>15.1f} "
            f"{gpu_neg1.get('max_gpu_util', 0):>15.1f} "
            f"{gpu_0.get('max_gpu_util', 0) - gpu_neg1.get('max_gpu_util', 0):>14.1f}"
        )
        print(
            f"{'Avg mem util (%)' :<30} {gpu_0.get('avg_mem_util', 0):>15.1f} "
            f"{gpu_neg1.get('avg_mem_util', 0):>15.1f} "
            f"{gpu_0.get('avg_mem_util', 0) - gpu_neg1.get('avg_mem_util', 0):>14.1f}"
        )

    print(f"\n{'='*80}")
    print("Conclusion:")
    print(f"{'='*80}")

    time_0 = results_0["get_tensor_time"]
    time_neg1 = results_neg1["get_tensor_time"]

    if time_0 > time_neg1:
        slowdown = (time_0 / time_neg1 - 1) * 100
        print(f"queue_size=0 is {slowdown:.1f}% slower than queue_size=-1")
        print(f"\nPossible causes:")
        print(f"  1. GPU memory bandwidth contention:")
        print(f"     - queue_size=0: producer and consumer access GPU concurrently")
        print(f"     - queue_size=-1: consumer has exclusive GPU access")
        print(f"  2. CUDA stream scheduling overhead")
        print(f"  3. Frequent CPU-GPU synchronization")

        if gpu_0.get("avg_gpu_util", 0) > gpu_neg1.get("avg_gpu_util", 0):
            print(f"\n  ✓ Confirmed: queue_size=0 has higher GPU util but worse perf")
            print(f"    Indicates resource contention, not under-utilization")

    print(f"{'='*80}\n")


def main():

    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Diagnose GPU resource contention")
    parser.add_argument("files", nargs="*", help="safetensors file paths")
    args = parser.parse_args()

    if args.files:
        test_files = []
        for pattern in args.files:
            test_files.extend(glob.glob(pattern))
        test_files = sorted(test_files)
    else:

        env_files = os.environ.get("TEST_FILES", "")
        if env_files:
            test_files = sorted(glob.glob(env_files))
        else:
            print("Error: please specify test files")
            print("Usage: python diagnose_gpu_contention.py /path/to/*.safetensors")
            sys.exit(1)

    if not test_files:
        print("Error: no test files found")
        sys.exit(1)

    print(f"Found {len(test_files)} test files")

    try:

        results_0 = test_with_profiling(test_files, queue_size=0)
        results_neg1 = test_with_profiling(test_files, queue_size=-1)

        compare_gpu_usage(results_0, results_neg1)

    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
