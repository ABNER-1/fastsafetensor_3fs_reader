#!/usr/bin/env python3
"""Shared utilities for USRBIO test scripts."""

import os
import time
import multiprocessing as mp
from typing import List, Dict, Callable

def worker_process_template(
    rank: int,
    file_paths: List[str],
    mount_point: str,
    buffer_size: int,
    chunk_size: int,
    num_iterations: int,
    results_queue: mp.Queue,
    verbose: bool = True,
    download_only: bool = False
):

    try:
        import torch
        from fastsafetensor_3fs_reader import ThreeFSFileReader
        
        if verbose:
            print(f"[Rank {rank}] Will read {len(file_paths)} files x {num_iterations} iterations")
        
        reader = ThreeFSFileReader(
            mount_point=mount_point,
            entries=64,
            io_depth=0,
            buffer_size=buffer_size
        )
        
        total_bytes_read = 0
        total_read_time = 0
        iteration_times = []
        
        for iteration in range(num_iterations):
            iteration_start = time.time()
            
            for file_idx, file_path in enumerate(file_paths):
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                
                start_read = time.time()
                if download_only:
                    bytes_read = reader.read_chunked(
                        path=file_path,
                        dev_ptr=0,
                        file_offset=0,
                        total_length=file_size,
                        chunk_size=chunk_size
                    )
                else:
                    device_id = rank % torch.cuda.device_count()
                    device = torch.device(f'cuda:{device_id}')
                    gpu_buffer = torch.empty(file_size, dtype=torch.uint8, device=device)
                    dev_ptr = gpu_buffer.data_ptr()
                    
                    bytes_read = reader.read_chunked(
                        path=file_path,
                        dev_ptr=dev_ptr,
                        file_offset=0,
                        total_length=file_size,
                        chunk_size=chunk_size
                    )
                    
                    del gpu_buffer
                    torch.cuda.synchronize()
                
                read_time = time.time() - start_read
                
                total_bytes_read += bytes_read
                total_read_time += read_time
                
                if verbose:
                    throughput_gbps = (bytes_read / 1024 / 1024 / 1024) / read_time
                    print(f"[Rank {rank}] Iter {iteration+1}/{num_iterations}, "
                          f"File {file_idx+1}/{len(file_paths)}: {file_name} - {throughput_gbps:.2f} GB/s")
            
            iteration_time = time.time() - iteration_start
            iteration_times.append(iteration_time)
            
            if verbose:
                print(f"[Rank {rank}] Iteration {iteration+1}/{num_iterations} completed in {iteration_time:.2f}s")
        
        del reader
        if not download_only:
            torch.cuda.synchronize()
        
        total_time = sum(iteration_times)
        avg_throughput_mbps = (total_bytes_read / 1024 / 1024) / total_time
        avg_throughput_gbps = avg_throughput_mbps / 1024
        
        result = {
            'rank': rank,
            'num_files': len(file_paths),
            'num_iterations': num_iterations,
            'total_bytes_read': total_bytes_read,
            'total_read_time': total_read_time,
            'total_time': total_time,
            'iteration_times': iteration_times,
            'avg_throughput_mbps': avg_throughput_mbps,
            'avg_throughput_gbps': avg_throughput_gbps,
            'success': True
        }
        
        if verbose:
            print(f"[Rank {rank}] Completed all iterations: {avg_throughput_gbps:.2f} GB/s average")
        
        results_queue.put(result)
        
    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
        
        result = {
            'rank': rank,
            'success': False,
            'error': str(e)
        }
        results_queue.put(result)

def assign_files_to_processes(files: List[str], num_processes: int) -> List[List[str]]:
    """Assign files to processes using stride pattern.

    Example: files=[f0..f7], num_processes=3 -> [[f0,f3,f6],[f1,f4,f7],[f2,f5]]
    """
    process_files = []
    for rank in range(num_processes):
        rank_files = [files[i] for i in range(rank, len(files), num_processes)]
        if rank_files:
            process_files.append(rank_files)
    
    return process_files

def run_concurrent_test(
    files: List[str],
    mount_point: str,
    num_processes: int,
    buffer_size: int,
    chunk_size: int,
    num_iterations: int,
    worker_func: Callable = None,
    verbose: bool = True,
    download_only: bool = False
) -> List[Dict]:

    if worker_func is None:
        worker_func = worker_process_template
    
    process_files = assign_files_to_processes(files, num_processes)
    actual_num_processes = len(process_files)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Starting concurrent test with {actual_num_processes} processes")
        print(f"{'='*80}")
        print(f"Mount point: {mount_point}")
        print(f"Buffer size: {buffer_size / 1024 / 1024:.0f} MB")
        print(f"Chunk size: {chunk_size / 1024 / 1024:.0f} MB")
        print(f"Iterations per file: {num_iterations}")
        print(f"Total files: {len(files)}")
        print(f"Test mode: {'DOWNLOAD ONLY (no GPU copy)' if download_only else 'FULL (download + GPU copy)'}")
        print(f"\nFile assignment (stride={actual_num_processes}):")
        for rank, rank_files in enumerate(process_files):
            file_indices = [i for i in range(rank, len(files), actual_num_processes)][:len(rank_files)]
            print(f"  Rank {rank}: {len(rank_files)} files - indices {file_indices}")
        print(f"{'='*80}\n")
    
    results_queue = mp.Queue()
    
    processes = []
    start_time = time.time()
    
    for rank, rank_files in enumerate(process_files):
        p = mp.Process(
            target=worker_func,
            args=(rank, rank_files, mount_point, buffer_size, chunk_size, num_iterations, results_queue, verbose, download_only)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    
    results.sort(key=lambda x: x.get('rank', 999))
    
    for r in results:
        r['wall_time'] = total_time
    
    return results

def print_test_results(results: List[Dict], num_iterations: int, total_files: int, verbose: bool = True):

    if not verbose:
        return
    
    print(f"\n{'='*80}")
    print("Test Results")
    print(f"{'='*80}\n")
    
    successful_results = [r for r in results if r.get('success', False)]
    failed_results = [r for r in results if not r.get('success', False)]
    
    if successful_results:
        print(f"{'Rank':<6} {'Files':<8} {'Iters':<8} {'Total(GB)':<12} {'Time(s)':<10} {'Throughput':<20}")
        print(f"{'-'*80}")
        
        total_bytes = 0
        wall_time = successful_results[0].get('wall_time', 0)
        
        for r in successful_results:
            total_gb = r['total_bytes_read'] / 1024 / 1024 / 1024
            print(f"{r['rank']:<6} {r['num_files']:<8} {r['num_iterations']:<8} {total_gb:<12.2f} "
                  f"{r['total_time']:<10.2f} {r['avg_throughput_mbps']:<10.2f} MB/s "
                  f"({r['avg_throughput_gbps']:.2f} GB/s)")
            total_bytes += r['total_bytes_read']
        
        print(f"{'-'*80}")
        
        aggregate_throughput_mbps = (total_bytes / 1024 / 1024) / wall_time
        aggregate_throughput_gbps = aggregate_throughput_mbps / 1024
        
        theoretical_max_mbps = sum(r['avg_throughput_mbps'] for r in successful_results)
        theoretical_max_gbps = theoretical_max_mbps / 1024
        
        print(f"\nAggregate Performance:")
        print(f"  Total data read: {total_bytes / 1024 / 1024 / 1024:.2f} GB")
        print(f"  Total wall time: {wall_time:.2f} seconds")
        print(f"  Aggregate throughput: {aggregate_throughput_mbps:.2f} MB/s ({aggregate_throughput_gbps:.2f} GB/s)")
        print(f"  Theoretical max (sum of all): {theoretical_max_mbps:.2f} MB/s ({theoretical_max_gbps:.2f} GB/s)")
        print(f"  Efficiency: {(aggregate_throughput_mbps / theoretical_max_mbps * 100):.1f}%")
        
        avg_throughput_mbps = sum(r['avg_throughput_mbps'] for r in successful_results) / len(successful_results)
        avg_throughput_gbps = avg_throughput_mbps / 1024
        print(f"\nAverage per-process performance:")
        print(f"  {avg_throughput_mbps:.2f} MB/s ({avg_throughput_gbps:.2f} GB/s)")
        
        all_iteration_times = []
        for r in successful_results:
            if 'iteration_times' in r:
                all_iteration_times.extend(r['iteration_times'])
        
        if all_iteration_times:
            avg_iter_time = sum(all_iteration_times) / len(all_iteration_times)
            min_iter_time = min(all_iteration_times)
            max_iter_time = max(all_iteration_times)
            print(f"\nIteration time statistics:")
            print(f"  Average: {avg_iter_time:.2f}s")
            print(f"  Min: {min_iter_time:.2f}s")
            print(f"  Max: {max_iter_time:.2f}s")
        
        print(f"\nTotal iterations: {num_iterations} x {total_files} files = {num_iterations * total_files} reads")
    
    if failed_results:
        print(f"\nFailed processes: {len(failed_results)}")
        for r in failed_results:
            print(f"  Rank {r['rank']}: {r.get('error', 'Unknown error')}")
    
    print(f"\n{'='*80}\n")

def print_summary_statistics(all_results: List[List[Dict]], num_repeats: int):

    if num_repeats <= 1:
        return
    
    print(f"\n{'='*80}")
    print(f"Summary of {num_repeats} iterations")
    print(f"{'='*80}\n")
    
    aggregate_throughputs = []
    for results in all_results:
        successful = [r for r in results if r.get('success', False)]
        if successful:
            total_bytes = sum(r['total_bytes_read'] for r in successful)
            wall_time = successful[0].get('wall_time', 1)
            throughput_gbps = (total_bytes / 1024 / 1024 / 1024) / wall_time
            aggregate_throughputs.append(throughput_gbps)
    
    if aggregate_throughputs:
        print(f"Aggregate throughput per iteration:")
        for i, tp in enumerate(aggregate_throughputs):
            print(f"  Iteration {i+1}: {tp:.2f} GB/s")
        
        avg_tp = sum(aggregate_throughputs) / len(aggregate_throughputs)
        min_tp = min(aggregate_throughputs)
        max_tp = max(aggregate_throughputs)
        
        print(f"\nStatistics:")
        print(f"  Average: {avg_tp:.2f} GB/s")
        print(f"  Min: {min_tp:.2f} GB/s")
        print(f"  Max: {max_tp:.2f} GB/s")
        print(f"  Variance: {max_tp - min_tp:.2f} GB/s")
    
    print(f"\n{'='*80}\n")
