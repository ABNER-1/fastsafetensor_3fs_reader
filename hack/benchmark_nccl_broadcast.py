#!/usr/bin/env python3
"""NCCL Broadcast Performance Benchmark.

Each GPU prepares 4 GB of random data (8 GPUs = 32 GB total).
Each round broadcasts all data in chunks of the specified block size; repeated 21 times.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist

def parse_size(size_str: str) -> int:
    size_str = size_str.upper()
    if size_str.endswith('M'):
        return int(size_str[:-1]) * 1024 * 1024
    elif size_str.endswith('G'):
        return int(size_str[:-1]) * 1024 * 1024 * 1024
    else:
        return int(size_str)

def format_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024**3):.2f} GB"
    elif size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024**2):.2f} MB"
    else:
        return f"{size_bytes / 1024:.2f} KB"

def format_bandwidth(bytes_per_sec: float) -> str:
    return f"{bytes_per_sec / (1024**3):.2f} GB/s"

class NCCLBroadcastBenchmark:
    
    def __init__(self, num_rounds: int = 21, warmup_rounds: int = 1):

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.num_rounds = num_rounds
        self.warmup_rounds = warmup_rounds
        
        if self.world_size != 8:
            if self.rank == 0:
                print(f"Error: need 8 GPUs, got {self.world_size}")
            sys.exit(1)
        
        torch.cuda.set_device(self.rank)
        self.device = torch.device(f"cuda:{self.rank}")
        
        self.data_per_gpu = 4 * 1024 * 1024 * 1024  # 4GB
        
        if self.rank == 0:
            print(f"\n{'='*80}")
            print(f"NCCL Broadcast Performance Benchmark")
            print(f"{'='*80}")
            print(f"GPU Count: {self.world_size}")
            print(f"Data per GPU: {format_size(self.data_per_gpu)}")
            print(f"Total Data: {format_size(self.data_per_gpu * self.world_size)}")
            print(f"Broadcast Rounds: {self.num_rounds}")
            print(f"Warmup Rounds: {self.warmup_rounds}")
            print(f"\nPreparing data on each GPU...")
        
        # 4 GB of float32 data (4 bytes/element)
        num_elements = self.data_per_gpu // 4
        self.data = torch.randn(num_elements, dtype=torch.float32, device=self.device)
        
        dist.barrier()
        if self.rank == 0:
            print(f"[Rank {self.rank}] Generated {format_size(self.data.numel() * 4)} random data")
        
        dist.barrier()
    
    def run_benchmark(self, block_size: int) -> Dict:

        blocks_per_gpu = self.data_per_gpu // block_size
        block_elements = block_size // 4  # float32: 4 bytes/element
        
        total_broadcasts = self.world_size * blocks_per_gpu
        
        if self.rank == 0:
            print(f"\n{'='*80}")
            print(f"Testing {format_size(block_size)} blocks...")
            print(f"Blocks per GPU: {blocks_per_gpu}")
            print(f"Broadcasts per round: {total_broadcasts}")
            print(f"{'='*80}\n")
        
        # Warmup
        if self.rank == 0:
            print(f"Warmup: {self.warmup_rounds} rounds...")
        
        for _ in range(self.warmup_rounds):
            self._run_single_round(block_size, block_elements, blocks_per_gpu)
        
        dist.barrier()
        
        if self.rank == 0:
            print(f"\nStarting benchmark...")
        
        round_times = []
        
        for round_idx in range(self.num_rounds):
            round_time = self._run_single_round(block_size, block_elements, blocks_per_gpu)
            round_times.append(round_time)
            
            if self.rank == 0:
                total_data = self.data_per_gpu * self.world_size  # 32GB
                bandwidth = total_data / round_time
                
                print(f"  Round {round_idx + 1}/{self.num_rounds}: "
                      f"{round_time * 1000:.2f} ms "
                      f"({total_broadcasts} broadcasts, "
                      f"{format_bandwidth(bandwidth)} total)")
        
        if self.rank == 0:
            results = self._compute_statistics(round_times, block_size, total_broadcasts)
            return results
        
        return {}
    
    def _run_single_round(self, block_size: int, block_elements: int, blocks_per_gpu: int) -> float:

        dist.barrier()
        
        recv_buffers = []
        
        if self.rank == 0:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        for gpu_id in range(self.world_size):
            for block_id in range(blocks_per_gpu):
                if self.rank == gpu_id:
                    # sender: use local data
                    start_idx = block_id * block_elements
                    end_idx = start_idx + block_elements
                    block_data = self.data[start_idx:end_idx].contiguous()
                else:
                    # receiver: allocate new GPU buffer
                    block_data = torch.empty(block_elements, dtype=torch.float32, device=self.device)
                    recv_buffers.append(block_data)
                
                # Broadcast
                dist.broadcast(block_data, src=gpu_id)
        
        if self.rank == 0:
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # ms -> s
        else:
            elapsed_time = 0.0
        
        del recv_buffers
        torch.cuda.empty_cache()
        
        return elapsed_time
    
    def _compute_statistics(self, round_times: List[float], block_size: int,
                          total_broadcasts: int) -> Dict:
        import statistics
        
        total_data = self.data_per_gpu * self.world_size  # 32GB
        
        avg_time = statistics.mean(round_times)
        min_time = min(round_times)
        max_time = max(round_times)
        std_time = statistics.stdev(round_times) if len(round_times) > 1 else 0.0
        
        avg_bandwidth = total_data / avg_time
        peak_bandwidth = total_data / min_time
        
        single_broadcast_bandwidth = block_size / (avg_time / total_broadcasts)
        
        return {
            'block_size': block_size,
            'broadcasts_per_round': total_broadcasts,
            'avg_time_ms': avg_time * 1000,
            'min_time_ms': min_time * 1000,
            'max_time_ms': max_time * 1000,
            'std_time_ms': std_time * 1000,
            'avg_bandwidth_gbps': avg_bandwidth / (1024**3),
            'peak_bandwidth_gbps': peak_bandwidth / (1024**3),
            'single_broadcast_bandwidth_gbps': single_broadcast_bandwidth / (1024**3),
            'round_times': round_times
        }
    
    def print_summary(self, all_results: List[Dict]):
        if self.rank != 0:
            return
        
        print(f"\n{'='*80}")
        print(f"Results Summary")
        print(f"{'='*80}\n")
        
        print(f"┌{'─'*10}┬{'─'*12}┬{'─'*13}┬{'─'*14}┬{'─'*10}┐")
        print(f"│ {'Block':<8} │ {'Broadcasts':<10} │ {'Avg Round':<11} │ {'Total BW':<12} │ {'Single':<8} │")
        print(f"│ {'Size':<8} │ {'per Round':<10} │ {'Time':<11} │ {'(GB/s)':<12} │ {'BW':<8} │")
        print(f"├{'─'*10}┼{'─'*12}┼{'─'*13}┼{'─'*14}┼{'─'*10}┤")
        
        for result in all_results:
            block_size_str = format_size(result['block_size']).replace(' ', '')
            broadcasts = result['broadcasts_per_round']
            avg_time = result['avg_time_ms']
            total_bw = result['avg_bandwidth_gbps']
            single_bw = result['single_broadcast_bandwidth_gbps']
            
            print(f"│ {block_size_str:<8} │ {broadcasts:<10} │ {avg_time:>9.1f} ms │ "
                  f"{total_bw:>10.1f} GB/s │ {single_bw:>6.1f} GB/s│")
        
        print(f"└{'─'*10}┴{'─'*12}┴{'─'*13}┴{'─'*14}┴{'─'*10}┘")
        
        best_bandwidth = max(r['peak_bandwidth_gbps'] for r in all_results)
        print(f"\nNVLink Bandwidth Analysis:")
        print(f"- Best observed: {best_bandwidth:.1f} GB/s")
        print(f"- Note: This is aggregate bandwidth across all GPUs")
    
    def save_results(self, all_results: List[Dict], output_file: str = "nccl_broadcast_results.csv"):
        if self.rank != 0:
            return
        
        import csv
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            writer.writerow([
                'Block Size (bytes)', 'Block Size', 'Broadcasts per Round',
                'Round', 'Time (ms)', 'Total Bandwidth (GB/s)'
            ])
            
            for result in all_results:
                block_size = result['block_size']
                block_size_str = format_size(block_size)
                broadcasts = result['broadcasts_per_round']
                
                for round_idx, round_time in enumerate(result['round_times']):
                    bandwidth = (self.data_per_gpu * self.world_size) / round_time / (1024**3)
                    writer.writerow([
                        block_size,
                        block_size_str,
                        broadcasts,
                        round_idx + 1,
                        round_time * 1000,
                        f"{bandwidth:.2f}"
                    ])
        
        print(f"\nResults saved to: {output_file}")
    
    def cleanup(self):
        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='NCCL Broadcast Performance Benchmark')
    parser.add_argument('--size', type=str, default='all',
                       help='Block size to test (512M, 1G, 2G, 4G, or "all")')
    parser.add_argument('--rounds', type=int, default=21,
                       help='Number of test rounds (default: 21)')
    parser.add_argument('--warmup', type=int, default=1,
                       help='Number of warmup rounds (default: 1)')
    parser.add_argument('--output', type=str, default='nccl_broadcast_results.csv',
                       help='Output CSV file (default: nccl_broadcast_results.csv)')
    
    args = parser.parse_args()
    
    benchmark = NCCLBroadcastBenchmark(num_rounds=args.rounds, warmup_rounds=args.warmup)
    
    if args.size.lower() == 'all':
        test_sizes = ['512M', '1G', '2G', '4G']
    else:
        test_sizes = [args.size]
    
    all_results = []
    
    for size_str in test_sizes:
        block_size = parse_size(size_str)
        result = benchmark.run_benchmark(block_size)
        if result:
            all_results.append(result)
    
    if all_results:
        benchmark.print_summary(all_results)
        benchmark.save_results(all_results, args.output)
    
    benchmark.cleanup()
    
    if benchmark.rank == 0:
        print(f"\n{'='*80}")
        print("Benchmark completed!")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
