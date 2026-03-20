
#!/usr/bin/env python3
"""Benchmark worker process for multi-backend performance testing.

This module provides the worker function that runs inside each benchmark
subprocess. It supports mock/python/cpp backends via
``fastsafetensor_3fs_reader.create_reader``.
"""

import os
import sys
import time
import multiprocessing as mp
from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# sys.path setup
#
# Do NOT do any package imports here at module level.  This module is loaded
# by multiprocessing.spawn child processes, and importing heavy packages
# (fastsafetensor_3fs_reader, torch, etc.) at module level would trigger
# CUDA / shared-library initialisation before the worker function runs,
# causing errors like "libcusparse.so: undefined symbol" in environments
# where LD_LIBRARY_PATH is not yet fully configured.
#
# We only manipulate sys.path here (pure string operations, no imports).
# The actual package imports happen inside benchmark_worker_process() and
# _create_reader_for_backend(), mirroring the pattern used by
# test_usrbio_common.py which works correctly.
#
# sys.path rule: NEVER insert the project root at position 0.
# Doing so would shadow the installed fastsafetensor_3fs_reader package
# (site-packages) with the raw source tree, which lacks _core_v2.so.
# The spawn child inherits sys.path from the parent, so a position-0
# insert in benchmark_runner.py would propagate here and break cpp imports.
# ---------------------------------------------------------------------------
_HACK_DIR = os.path.dirname(os.path.abspath(__file__))
if _HACK_DIR not in sys.path:
    sys.path.append(_HACK_DIR)


def benchmark_worker_process(
    rank: int,
    file_paths: List[str],
    mount_point: str,
    backend: str,
    buffer_size: int,
    chunk_size: int,
    num_processes: int,
    num_iterations: int,
    results_queue: mp.Queue,
    verbose: bool = True,
    download_only: bool = False,
) -> None:
    """Worker process that reads files using the specified backend and reports
    performance metrics.

    Args:
        rank: Process rank (0-based).
        file_paths: List of file paths assigned to this worker.
        mount_point: 3FS FUSE mount-point path.
        backend: Reader backend name (``"mock"``, ``"python"``, ``"cpp"``).
        buffer_size: IOV buffer size in bytes.
        chunk_size: Per-read chunk size in bytes.
        num_processes: Total number of concurrent processes (for metadata).
        num_iterations: Number of iterations over the file list.
        results_queue: ``mp.Queue`` to push per-iteration result dicts.
        verbose: If *True*, print per-file progress.
        download_only: If *True*, read into host memory only (dev_ptr=0).
    """
    try:
        import torch
        from fastsafetensor_3fs_reader import create_reader

        # ---- create reader ------------------------------------------------
        reader = _create_reader_for_backend(
            backend=backend,
            mount_point=mount_point,
            buffer_size=buffer_size,
        )

        if verbose:
            label = f"[Rank {rank} | {backend}]"
            print(
                f"{label} Will read {len(file_paths)} files x "
                f"{num_iterations} iterations  "
                f"(buffer={buffer_size // (1024 * 1024)}MB, "
                f"chunk={chunk_size // (1024 * 1024)}MB)"
            )

        # ---- iterate -------------------------------------------------------
        for iteration in range(num_iterations):
            iter_bytes = 0
            iter_read_time = 0.0
            iter_start = time.time()

            for file_idx, file_path in enumerate(file_paths):
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)

                read_start = time.time()

                if download_only:
                    bytes_read = reader.read_chunked(
                        path=file_path,
                        dev_ptr=0,
                        file_offset=0,
                        total_length=file_size,
                        chunk_size=chunk_size,
                    )
                else:
                    device_id = rank % torch.cuda.device_count()
                    device = torch.device(f"cuda:{device_id}")
                    gpu_buffer = torch.empty(
                        file_size, dtype=torch.uint8, device=device
                    )
                    dev_ptr = gpu_buffer.data_ptr()

                    bytes_read = reader.read_chunked(
                        path=file_path,
                        dev_ptr=dev_ptr,
                        file_offset=0,
                        total_length=file_size,
                        chunk_size=chunk_size,
                    )

                    del gpu_buffer
                    torch.cuda.synchronize()

                read_time = time.time() - read_start
                iter_bytes += bytes_read
                iter_read_time += read_time

                if verbose:
                    throughput_gbps = (
                        (bytes_read / 1024 / 1024 / 1024) / read_time
                        if read_time > 0
                        else 0.0
                    )
                    print(
                        f"[Rank {rank} | {backend}] "
                        f"Iter {iteration + 1}/{num_iterations}, "
                        f"File {file_idx + 1}/{len(file_paths)}: "
                        f"{file_name} - {throughput_gbps:.2f} GB/s"
                    )

            iter_wall = time.time() - iter_start

            throughput_mbps = (
                (iter_bytes / 1024 / 1024) / iter_wall if iter_wall > 0 else 0.0
            )
            throughput_gbps = throughput_mbps / 1024

            result: Dict[str, Any] = {
                "backend": backend,
                "buffer_size": buffer_size,
                "chunk_size": chunk_size,
                "num_processes": num_processes,
                "rank": rank,
                "iteration": iteration,
                "num_files": len(file_paths),
                "total_bytes_read": iter_bytes,
                "read_time": iter_read_time,
                "wall_time": iter_wall,
                "throughput_mbps": throughput_mbps,
                "throughput_gbps": throughput_gbps,
                "success": True,
                "error": None,
            }
            results_queue.put(result)

            if verbose:
                print(
                    f"[Rank {rank} | {backend}] "
                    f"Iteration {iteration + 1}/{num_iterations} completed: "
                    f"{throughput_gbps:.2f} GB/s  "
                    f"(wall={iter_wall:.2f}s, io={iter_read_time:.2f}s)"
                )

        # ---- cleanup -------------------------------------------------------
        reader.close()
        if not download_only:
            torch.cuda.synchronize()

    except Exception as e:
        import traceback

        print(f"[Rank {rank} | {backend}] Error: {e}")
        traceback.print_exc()

        error_result: Dict[str, Any] = {
            "backend": backend,
            "buffer_size": buffer_size,
            "chunk_size": chunk_size,
            "num_processes": num_processes,
            "rank": rank,
            "iteration": -1,
            "num_files": len(file_paths),
            "total_bytes_read": 0,
            "read_time": 0.0,
            "wall_time": 0.0,
            "throughput_mbps": 0.0,
            "throughput_gbps": 0.0,
            "success": False,
            "error": str(e),
        }
        results_queue.put(error_result)


def _create_reader_for_backend(
    backend: str,
    mount_point: str,
    buffer_size: int,
) -> Any:
    """Instantiate a FileReaderInterface for the given backend.

    MockFileReader only accepts ``mount_point``; the other backends also
    accept ``entries``, ``io_depth``, and ``buffer_size``.
    """
    from fastsafetensor_3fs_reader import create_reader

    if backend == "mock":
        # MockFileReader.__init__ signature: (mount_point: str = "")
        return create_reader(backend="mock", mount_point=mount_point)
    else:
        return create_reader(
            backend=backend,
            mount_point=mount_point,
            entries=64,
            io_depth=0,
            buffer_size=buffer_size,
        )


def assign_files_to_processes(
    files: List[str], num_processes: int
) -> List[List[str]]:
    """Distribute *files* across *num_processes* workers using a stride pattern.

    Example::

        files = [f0, f1, f2, f3, f4, f5, f6, f7]
        num_processes = 3
        -> [[f0, f3, f6], [f1, f4, f7], [f2, f5]]
    """
    process_files: List[List[str]] = []
    for rank in range(num_processes):
        rank_files = [files[i] for i in range(rank, len(files), num_processes)]
        if rank_files:
            process_files.append(rank_files)
    return process_files


def run_benchmark_round(
    files: List[str],
    mount_point: str,
    backend: str,
    num_processes: int,
    buffer_size: int,
    chunk_size: int,
    num_iterations: int,
    verbose: bool = True,
    download_only: bool = False,
) -> List[Dict[str, Any]]:
    """Run a single benchmark round: spawn workers, collect results.

    Returns a list of per-iteration result dicts from all workers.
    """
    process_files = assign_files_to_processes(files, num_processes)
    actual_num_processes = len(process_files)

    if verbose:
        print(
            f"\n--- Round: backend={backend}, "
            f"buffer={buffer_size // (1024 * 1024)}MB, "
            f"chunk={chunk_size // (1024 * 1024)}MB, "
            f"procs={actual_num_processes} ---"
        )

    results_queue: mp.Queue = mp.Queue()
    processes: List[mp.Process] = []

    round_start = time.time()

    for rank, rank_files in enumerate(process_files):
        p = mp.Process(
            target=benchmark_worker_process,
            args=(
                rank,
                rank_files,
                mount_point,
                backend,
                buffer_size,
                chunk_size,
                actual_num_processes,
                num_iterations,
                results_queue,
                verbose,
                download_only,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    round_wall = time.time() - round_start

    results: List[Dict[str, Any]] = []
    while not results_queue.empty():
        results.append(results_queue.get())

    # Attach the overall wall-clock time to every result record
    for r in results:
        r["round_wall_time"] = round_wall

    return results


# ---------------------------------------------------------------------------
# headers_batch benchmark
# ---------------------------------------------------------------------------

def benchmark_headers_batch_worker_process(
    rank: int,
    file_paths: List[str],
    mount_point: str,
    backend: str,
    buffer_size: int,
    batch_size: int,
    num_threads: int,
    num_processes: int,
    num_iterations: int,
    results_queue: mp.Queue,
    verbose: bool = True,
) -> None:
    """Worker process that benchmarks ``read_headers_batch`` for the given backend.

    Splits *file_paths* into batches of *batch_size* and calls
    ``reader.read_headers_batch(batch, num_threads)`` for each batch.
    Metrics reported per iteration:

    * ``total_files_read``  – number of files whose headers were read
    * ``headers_per_second`` – aggregate files/s across all batches
    * ``wall_time``          – wall-clock time for the full iteration

    Args:
        rank: Process rank (0-based).
        file_paths: Files assigned to this worker.
        mount_point: 3FS FUSE mount-point path.
        backend: Reader backend name (``"python"`` or ``"cpp"``).
        buffer_size: IOV buffer size in bytes (passed to reader constructor).
        batch_size: Number of files per ``read_headers_batch`` call.
        num_threads: ``num_threads`` argument forwarded to ``read_headers_batch``.
        num_processes: Total concurrent processes (for metadata only).
        num_iterations: Number of full passes over *file_paths*.
        results_queue: ``mp.Queue`` to push per-iteration result dicts.
        verbose: If *True*, print per-iteration progress.
    """
    try:
        reader = _create_reader_for_backend(
            backend=backend,
            mount_point=mount_point,
            buffer_size=buffer_size,
        )

        if verbose:
            label = f"[Rank {rank} | {backend} | headers_batch]"
            print(
                f"{label} Will read {len(file_paths)} files x "
                f"{num_iterations} iterations  "
                f"(batch_size={batch_size}, num_threads={num_threads})"
            )

        for iteration in range(num_iterations):
            iter_files = 0
            iter_start = time.time()

            # Split file_paths into batches of batch_size
            batches = [
                file_paths[i : i + batch_size]
                for i in range(0, len(file_paths), batch_size)
            ]

            for batch_idx, batch in enumerate(batches):
                batch_start = time.time()
                results = reader.read_headers_batch(batch, num_threads)
                batch_elapsed = time.time() - batch_start
                iter_files += len(results)

                if verbose:
                    batch_fps = len(results) / batch_elapsed if batch_elapsed > 0 else 0.0
                    print(
                        f"[Rank {rank} | {backend} | headers_batch] "
                        f"Iter {iteration + 1}/{num_iterations}, "
                        f"Batch {batch_idx + 1}/{len(batches)}: "
                        f"{len(results)} files in {batch_elapsed:.3f}s "
                        f"({batch_fps:.1f} files/s)"
                    )

            iter_wall = time.time() - iter_start
            headers_per_second = iter_files / iter_wall if iter_wall > 0 else 0.0

            result: Dict[str, Any] = {
                "op": "headers_batch",
                "backend": backend,
                "buffer_size": buffer_size,
                "batch_size": batch_size,
                "num_threads": num_threads,
                "num_processes": num_processes,
                "rank": rank,
                "iteration": iteration,
                "num_files": len(file_paths),
                "total_files_read": iter_files,
                "headers_per_second": headers_per_second,
                "wall_time": iter_wall,
                "success": True,
                "error": None,
            }
            results_queue.put(result)

            if verbose:
                print(
                    f"[Rank {rank} | {backend} | headers_batch] "
                    f"Iteration {iteration + 1}/{num_iterations} completed: "
                    f"{headers_per_second:.1f} files/s  "
                    f"(wall={iter_wall:.2f}s, files={iter_files})"
                )

        reader.close()

    except Exception as e:
        import traceback

        print(f"[Rank {rank} | {backend} | headers_batch] Error: {e}")
        traceback.print_exc()

        error_result: Dict[str, Any] = {
            "op": "headers_batch",
            "backend": backend,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "num_threads": num_threads,
            "num_processes": num_processes,
            "rank": rank,
            "iteration": -1,
            "num_files": len(file_paths),
            "total_files_read": 0,
            "headers_per_second": 0.0,
            "wall_time": 0.0,
            "success": False,
            "error": str(e),
        }
        results_queue.put(error_result)


def run_headers_batch_round(
    files: List[str],
    mount_point: str,
    backend: str,
    num_processes: int,
    buffer_size: int,
    batch_size: int,
    num_threads: int,
    num_iterations: int,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run a single headers_batch benchmark round: spawn workers, collect results.

    Each worker receives a strided subset of *files* and calls
    ``read_headers_batch`` in batches of *batch_size*.

    Returns a list of per-iteration result dicts from all workers.
    """
    process_files = assign_files_to_processes(files, num_processes)
    actual_num_processes = len(process_files)

    if verbose:
        print(
            f"\n--- Round [headers_batch]: backend={backend}, "
            f"batch_size={batch_size}, num_threads={num_threads}, "
            f"procs={actual_num_processes} ---"
        )

    results_queue: mp.Queue = mp.Queue()
    processes: List[mp.Process] = []

    round_start = time.time()

    for rank, rank_files in enumerate(process_files):
        p = mp.Process(
            target=benchmark_headers_batch_worker_process,
            args=(
                rank,
                rank_files,
                mount_point,
                backend,
                buffer_size,
                batch_size,
                num_threads,
                actual_num_processes,
                num_iterations,
                results_queue,
                verbose,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    round_wall = time.time() - round_start

    results: List[Dict[str, Any]] = []
    while not results_queue.empty():
        results.append(results_queue.get())

    # Attach the overall wall-clock time to every result record
    for r in results:
        r["round_wall_time"] = round_wall

    return results
