#!/usr/bin/env python3
"""Benchmark queue_size=-1 vs queue_size=0 for ParallelLoader.

Usage:
    python test_queue_performance.py /path/to/safetensors/files/*.safetensors
    export TEST_FILES="/path/to/model-*.safetensors" && python test_queue_performance.py
"""

import os
import sys
import time
import glob
import argparse
from typing import List, Dict
from fastsafetensors import ParallelLoader

class PerformanceMetrics:

    def __init__(self):
        self.batch_metrics: List[Dict] = []
        self.total_time = 0
        self.tensor_count = 0
        
    def add_batch(self, batch_id: int, metrics: Dict):

        self.batch_metrics.append({
            'batch_id': batch_id,
            **metrics
        })
        
    def summary(self) -> Dict:

        if not self.batch_metrics:
            return {}
            
        total_add_filenames = sum(b.get('add_filenames_time', 0) for b in self.batch_metrics)
        total_copy_files = sum(b.get('copy_files_time', 0) for b in self.batch_metrics)
        total_get_tensor = sum(b.get('get_tensor_time', 0) for b in self.batch_metrics)
        total_close = sum(b.get('close_time', 0) for b in self.batch_metrics)
        
        return {
            'total_batches': len(self.batch_metrics),
            'total_time': self.total_time,
            'tensor_count': self.tensor_count,
            'avg_time_per_tensor': self.total_time / self.tensor_count if self.tensor_count > 0 else 0,
            'total_add_filenames_time': total_add_filenames,
            'total_copy_files_time': total_copy_files,
            'total_get_tensor_time': total_get_tensor,
            'total_close_time': total_close,
            'avg_add_filenames_per_batch': total_add_filenames / len(self.batch_metrics),
            'avg_copy_files_per_batch': total_copy_files / len(self.batch_metrics),
            'avg_get_tensor_per_batch': total_get_tensor / len(self.batch_metrics),
            'avg_close_per_batch': total_close / len(self.batch_metrics),
        }

def find_test_files() -> List[str]:

    if len(sys.argv) > 1:
        files = []
        for pattern in sys.argv[1:]:
            files.extend(glob.glob(pattern))
        if files:
            return sorted(files)
    
    env_files = os.environ.get('TEST_FILES', '')
    if env_files:
        files = glob.glob(env_files)
        if files:
            return sorted(files)
    
    patterns = [
        '*.safetensors',
        'model*.safetensors',
        'pytorch_model*.safetensors',
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return sorted(files)
    
raise FileNotFoundError(
        "No test files found. Specify via:\n"
        "1. CLI args: python test_queue_performance.py /path/to/*.safetensors\n"
        "2. Env var: export TEST_FILES='/path/to/*.safetensors'\n"
        "3. Place .safetensors files in the current directory"
    )

def test_queue_performance(test_files: List[str], queue_size: int, test_name: str) -> PerformanceMetrics:
    print(f"\n{'='*80}")
    print(f"Test: {test_name} (queue_size={queue_size})")
    print(f"{'='*80}")
    print(f"Test files: {len(test_files)}")
    for i, f in enumerate(test_files, 1):
        print(f"  {i}. {os.path.basename(f)}")
    print(f"{'='*80}\n")
    
    os.environ["FASTSAFETENSORS_DEBUG"] = "true"
    
    metrics = PerformanceMetrics()
    start_time = time.time()
    
    try:
        iterator = ParallelLoader(
            pg=None,
            hf_weights_files=test_files,
            max_concurrent_producers=1,
            queue_size=queue_size,
            use_tqdm_on_load=True,
            device="cuda:0",
            nogds=False,
            debug_log=True
        )
        
        tensor_count = 0
        for key, tensor in iterator.iterate_weights():
            tensor_count += 1
        
        iterator.close()
        
        metrics.tensor_count = tensor_count
        
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        end_time = time.time()
        metrics.total_time = end_time - start_time
    
    summary = metrics.summary()
    print(f"\n{'='*80}")
    print(f"Results: {test_name}")
    print(f"{'='*80}")
    print(f"Total time: {metrics.total_time:.2f}s")
    print(f"Tensor count: {metrics.tensor_count}")
    if metrics.tensor_count > 0:
        print(f"Avg per tensor: {metrics.total_time/metrics.tensor_count*1000:.2f} ms")
    print(f"{'='*80}\n")
    
    return metrics

def compare_results(metrics_0: PerformanceMetrics, metrics_neg1: PerformanceMetrics):
    print(f"\n{'='*80}")
    print("Performance comparison")
    print(f"{'='*80}\n")
    
    time_0 = metrics_0.total_time
    time_neg1 = metrics_neg1.total_time
    
    print(f"{'Metric':<30} {'queue_size=0':>15} {'queue_size=-1':>15} {'Diff':>15}")
    print(f"{'-'*80}")
    print(f"{'Total time (s)':<30} {time_0:>15.2f} {time_neg1:>15.2f} {(time_neg1/time_0-1)*100:>14.1f}%")
    print(f"{'Tensor count':<30} {metrics_0.tensor_count:>15} {metrics_neg1.tensor_count:>15} {'-':>15}")
    
    if metrics_0.tensor_count > 0:
        avg_0 = time_0 / metrics_0.tensor_count * 1000
        avg_neg1 = time_neg1 / metrics_neg1.tensor_count * 1000
        print(f"{'Avg per tensor (ms)':<30} {avg_0:>15.2f} {avg_neg1:>15.2f} {(avg_neg1/avg_0-1)*100:>14.1f}%")
    
    print(f"\n{'='*80}")
    print("Conclusion:")
    print(f"{'='*80}")
    
    if time_neg1 > time_0:
        slowdown = (time_neg1 / time_0 - 1) * 100
        print(f"queue_size=-1 is {slowdown:.1f}% slower than queue_size=0")
        print(f"\nReason:")
        print(f"  - queue_size=-1: producer must wait for consumer to finish (get_tensor + close)")
        print(f"  - queue_size=0:  producer can proceed once consumer dequeues the batch")
        print(f"  - Result: queue_size=0 achieves better pipeline parallelism")
    else:
        print(f"queue_size=0 is {(time_0/time_neg1-1)*100:.1f}% slower than queue_size=-1")
        print(f"Unexpected result; possible environment issue")
    
    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark ParallelLoader with different queue_size values',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_queue_performance.py /path/to/model-*.safetensors
  python test_queue_performance.py model-00001.safetensors model-00002.safetensors

  export TEST_FILES="/path/to/model-*.safetensors"
  python test_queue_performance.py
        """
    )
    parser.add_argument('files', nargs='*', help='safetensors file paths (glob patterns supported)')
    args = parser.parse_args()
    
    try:
        if args.files:
            test_files = []
            for pattern in args.files:
                test_files.extend(glob.glob(pattern))
            test_files = sorted(test_files)
        else:
            test_files = find_test_files()
        
        if not test_files:
            print("Error: no safetensors files found")
            sys.exit(1)
        
        print(f"Found {len(test_files)} test files")
        
        metrics_0 = test_queue_performance(
            test_files, 
            queue_size=0, 
            test_name="Queue Size = 0 (high-concurrency mode)"
        )
        
        metrics_neg1 = test_queue_performance(
            test_files,
            queue_size=-1,
            test_name="Queue Size = -1 (low-concurrency mode)"
        )
        
        compare_results(metrics_0, metrics_neg1)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
