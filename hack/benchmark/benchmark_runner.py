
#!/usr/bin/env python3
"""Benchmark runner: multi-dimensional parameter sweep for 3FS reader backends.

Supports two modes:
- **grid**: Exhaustive grid search over buffer_size x chunk_size x backend x
  num_processes.
- **single**: Run a single parameter combination for quick validation.

Usage examples::

    # Full grid search (default parameters)
    python benchmark_runner.py --mount-point /mnt/3fs

    # Custom grid
    python benchmark_runner.py --mount-point /mnt/3fs \\
        --backends cpp,python \\
        --buffer-sizes 256,512,1024 \\
        --chunk-sizes 32,64,128 \\
        --num-processes 4,8 \\
        --iterations 5

    # Single-config quick test
    python benchmark_runner.py --mount-point /mnt/3fs \\
        --mode single \\
        --backends mock \\
        --buffer-sizes 64 \\
        --chunk-sizes 8 \\
        --num-processes 1 \\
        --iterations 1

    # Download-only (no GPU copy)
    python benchmark_runner.py --mount-point /mnt/3fs --download-only
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# sys.path setup
#
# benchmark_worker and benchmark_report live in the same hack/ directory.
# We need them on sys.path so they can be imported here and also by the
# multiprocessing.spawn child processes (which inherit sys.path from the
# parent).
#
# IMPORTANT: do NOT insert the project root at position 0.  Doing so would
# shadow the *installed* fastsafetensor_3fs_reader package (in site-packages)
# with the raw source tree, which lacks the compiled _core_v2.so extension.
# The spawn child processes inherit sys.path, so a position-0 insert here
# propagates to every worker subprocess and causes ImportError there.
#
# Safe strategy:
#   - Add the hack/ directory (for benchmark_worker / benchmark_report).
#   - Add the project root only as a fallback if the package is not yet
#     importable (e.g. pure-dev environment without pip install).
# ---------------------------------------------------------------------------
_HACK_DIR = os.path.dirname(os.path.abspath(__file__))
if _HACK_DIR not in sys.path:
    sys.path.append(_HACK_DIR)

# fastsafetensor_3fs_reader is intentionally NOT imported here in the main
# process.  Importing it (especially with FASTSAFETENSORS_BACKEND=cpp) would
# load _core_v2.so and CUDA libraries (libcusparse, etc.) in the parent
# process.  Even though multiprocessing.spawn creates fresh child processes,
# loading CUDA libraries in the parent can interfere with how the dynamic
# linker resolves symbols in the children, causing errors like:
#   "libcusparse.so.12: undefined symbol: __nvJitLinkGetErrorLog_12_6"
# The package is imported lazily inside benchmark_worker_process() instead,
# exactly as test_usrbio_common.py does.

from benchmark_worker import run_benchmark_round, run_headers_batch_round
from benchmark_report import (
    aggregate_results,
    export_csv,
    generate_charts,
    print_console_report,
    aggregate_headers_batch_results,
    print_headers_batch_report,
    export_headers_batch_csv,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_int_list(value: str) -> List[int]:
    """Parse a comma-separated string of integers."""
    return [int(x.strip()) for x in value.split(",") if x.strip()]

def _parse_float_list(value: str) -> List[float]:
    """Parse a comma-separated string of numbers (int or float, e.g. '0.5,1,2,4')."""
    try:
        return [float(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid value {value!r}: expected comma-separated numbers (e.g. '0.5,1,2,4')"
        )


def _parse_str_list(value: str) -> List[str]:
    """Parse a comma-separated string of names."""
    return [x.strip() for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-dimensional benchmark for fastsafetensor_3fs_reader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--mount-point",
        required=True,
        help="3FS FUSE mount-point path (e.g. /mnt/3fs)",
    )

    # File selection
    parser.add_argument(
        "--file-pattern",
        default=None,
        help=(
            "Glob pattern for safetensors files. "
            "Default: <mount-point>/*.safetensors"
        ),
    )

    # Parameter space
    parser.add_argument(
        "--backends",
        type=_parse_str_list,
        default=["mock", "python", "cpp"],
        help="Comma-separated backend names (default: mock,python,cpp)",
    )
    parser.add_argument(
        "--buffer-sizes",
        type=_parse_float_list,
        default=[64, 128, 256, 512, 1024],
        help="Comma-separated buffer sizes in MB (default: 64,128,256,512,1024; supports floats e.g. 0.5,1,2)",
    )
    parser.add_argument(
        "--chunk-sizes",
        type=_parse_float_list,
        default=[64, 128, 256, 512, 1024],
        help=(
            "Comma-separated chunk sizes in MB "
            "(default: same as --buffer-sizes: 64,128,256,512,1024)"
        ),
    )
    parser.add_argument(
        "--no-equal-chunk-buffer",
        action="store_true",
        help=(
            "Allow chunk_size < buffer_size combinations. "
            "By default only chunk_size == buffer_size is tested, "
            "which keeps the parameter space small."
        ),
    )
    parser.add_argument(
        "--num-processes",
        type=_parse_int_list,
        default=[1, 2, 4, 8],
        help="Comma-separated process counts (default: 1,2,4,8)",
    )

    # headers_batch benchmark parameters
    parser.add_argument(
        "--benchmark-op",
        choices=["read_chunked", "headers_batch", "both"],
        default="read_chunked",
        help=(
            "Which operation to benchmark: read_chunked, headers_batch, or both "
            "(default: read_chunked)"
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        type=_parse_int_list,
        default=[8, 16, 32, 64, 128],
        help=(
            "Comma-separated file batch sizes for headers_batch benchmark "
            "(default: 8,16,32,64,128)"
        ),
    )
    parser.add_argument(
        "--num-threads",
        type=_parse_int_list,
        default=[4, 8, 16],
        help=(
            "Comma-separated num_threads values for headers_batch "
            "(default: 4,8,16)"
        ),
    )

    # Execution control
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Iterations per parameter combination (default: 3)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup rounds before measurement (default: 1)",
    )
    parser.add_argument(
        "--mode",
        choices=["grid", "single"],
        default="grid",
        help="Run mode: grid (sweep all combos) or single (default: grid)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Read into host memory only (skip GPU copy)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="./benchmark_results",
        help="Directory for CSV and chart output (default: ./benchmark_results)",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip matplotlib chart generation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print per-file progress (default: True)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file progress (overrides --verbose)",
    )

    return parser


# ---------------------------------------------------------------------------
# Parameter combination generation
# ---------------------------------------------------------------------------

def generate_combinations(
    backends: List[str],
    buffer_sizes_mb: List[float],
    chunk_sizes_mb: List[float],
    num_processes_list: List[int],
    equal_only: bool = True,
) -> List[Dict[str, Any]]:
    """Generate all valid (backend, buffer_size, chunk_size, num_processes)
    combinations.

    When *equal_only* is ``True`` (default), only combinations where
    ``chunk_size == buffer_size`` are included.  This keeps the parameter
    space small and is the recommended starting point.

    When *equal_only* is ``False``, all combinations where
    ``chunk_size <= buffer_size`` are included (full grid search).

    For the ``mock`` backend, buffer_size and chunk_size are not meaningful,
    so only one representative combination per num_processes is generated.
    """
    combos: List[Dict[str, Any]] = []

    for backend in backends:
        if backend == "mock":
            # Mock ignores buffer/chunk; use a single placeholder per nprocs
            for nprocs in num_processes_list:
                combos.append(
                    {
                        "backend": backend,
                        "buffer_size_mb": 0,
                        "chunk_size_mb": 0,
                        "num_processes": nprocs,
                    }
                )
        else:
            for buf_mb in buffer_sizes_mb:
                for chunk_mb in chunk_sizes_mb:
                    if equal_only and chunk_mb != buf_mb:
                        continue  # only test chunk == buffer
                    if not equal_only and chunk_mb > buf_mb:
                        continue  # invalid: chunk cannot exceed buffer
                    for nprocs in num_processes_list:
                        combos.append(
                            {
                                "backend": backend,
                                "buffer_size_mb": buf_mb,
                                "chunk_size_mb": chunk_mb,
                                "num_processes": nprocs,
                            }
                        )

    return combos


def generate_headers_batch_combinations(
    backends: List[str],
    batch_sizes: List[int],
    num_threads_list: List[int],
    num_processes_list: List[int],
) -> List[Dict[str, Any]]:
    """Generate all valid (backend, batch_size, num_threads, num_processes)
    combinations for the headers_batch benchmark.

    The ``mock`` backend is excluded because its ``read_headers_batch``
    performs no real I/O and is not meaningful to benchmark.
    """
    combos: List[Dict[str, Any]] = []
    for backend in backends:
        if backend == "mock":
            continue  # mock has no real I/O for headers_batch
        for batch_size in batch_sizes:
            for num_threads in num_threads_list:
                for nprocs in num_processes_list:
                    combos.append(
                        {
                            "backend": backend,
                            "batch_size": batch_size,
                            "num_threads": num_threads,
                            "num_processes": nprocs,
                        }
                    )
    return combos


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> None:
    """Execute the full benchmark according to parsed CLI arguments."""

    # ---- Discover files ----------------------------------------------------
    file_pattern = args.file_pattern or f"{args.mount_point}/*.safetensors"
    files = sorted(glob.glob(file_pattern))
    if not files:
        print(f"ERROR: No files found matching pattern: {file_pattern}")
        sys.exit(1)

    verbose = args.verbose and not args.quiet
    benchmark_op = args.benchmark_op  # "read_chunked", "headers_batch", or "both"

    print(f"\n{'#' * 90}")
    print(f"  3FS Reader Benchmark")
    print(f"{'#' * 90}")
    print(f"  Mount point   : {args.mount_point}")
    print(f"  File pattern  : {file_pattern}")
    print(f"  Files found   : {len(files)}")
    total_size = sum(os.path.getsize(f) for f in files)
    print(f"  Total size    : {total_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"  Backends      : {', '.join(args.backends)}")
    print(f"  Benchmark op  : {benchmark_op}")
    if benchmark_op in ("read_chunked", "both"):
        print(f"  Buffer sizes  : {args.buffer_sizes} MB")
        print(f"  Chunk sizes   : {args.chunk_sizes} MB")
    if benchmark_op in ("headers_batch", "both"):
        print(f"  Batch sizes   : {args.batch_sizes} files")
        print(f"  Num threads   : {args.num_threads}")
    print(f"  Num processes : {args.num_processes}")
    print(f"  Iterations    : {args.iterations}")
    print(f"  Warmup        : {args.warmup}")
    print(f"  Mode          : {args.mode}")
    if benchmark_op in ("read_chunked", "both"):
        print(f"  Download only : {args.download_only}")
    print(f"  Output dir    : {args.output_dir}")
    equal_only = not args.no_equal_chunk_buffer
    if benchmark_op in ("read_chunked", "both"):
        print(
            f"  Constraint    : "
            f"{'chunk == buffer (equal mode)' if equal_only else 'chunk <= buffer (full grid)'}"
        )
    print(f"{'#' * 90}\n")

    benchmark_start = time.time()

    # ==========================================================================
    # read_chunked benchmark
    # ==========================================================================
    if benchmark_op in ("read_chunked", "both"):
        combos = generate_combinations(
            backends=args.backends,
            buffer_sizes_mb=args.buffer_sizes,
            chunk_sizes_mb=args.chunk_sizes,
            num_processes_list=args.num_processes,
            equal_only=equal_only,
        )

        if not combos:
            print("WARNING: No valid read_chunked parameter combinations generated.")
        else:
            print(f"[read_chunked] Total parameter combinations: {len(combos)}")
            all_raw_results: List[Dict[str, Any]] = []

            for idx, combo in enumerate(combos, 1):
                backend = combo["backend"]
                buf_mb = combo["buffer_size_mb"]
                chunk_mb = combo["chunk_size_mb"]
                nprocs = combo["num_processes"]

                buffer_size = int(buf_mb * 1024 * 1024) if buf_mb > 0 else 64 * 1024 * 1024
                chunk_size = int(chunk_mb * 1024 * 1024) if chunk_mb > 0 else 64 * 1024 * 1024

                progress = f"[{idx}/{len(combos)}]"
                print(
                    f"\n{'=' * 90}\n"
                    f"{progress} [read_chunked] backend={backend}, "
                    f"buffer={buf_mb}MB, chunk={chunk_mb}MB, procs={nprocs}\n"
                    f"{'=' * 90}"
                )

                if args.warmup > 0:
                    print(f"  Warmup: {args.warmup} round(s)...")
                    for _w in range(args.warmup):
                        run_benchmark_round(
                            files=files,
                            mount_point=args.mount_point,
                            backend=backend,
                            num_processes=nprocs,
                            buffer_size=buffer_size,
                            chunk_size=chunk_size,
                            num_iterations=1,
                            verbose=False,
                            download_only=args.download_only,
                        )
                    print("  Warmup complete.")

                round_results = run_benchmark_round(
                    files=files,
                    mount_point=args.mount_point,
                    backend=backend,
                    num_processes=nprocs,
                    buffer_size=buffer_size,
                    chunk_size=chunk_size,
                    num_iterations=args.iterations,
                    verbose=verbose,
                    download_only=args.download_only,
                )

                all_raw_results.extend(round_results)

                successful = [r for r in round_results if r.get("success")]
                if successful:
                    total_bytes = sum(r["total_bytes_read"] for r in successful)
                    round_wall = successful[0].get(
                        "round_wall_time",
                        max(r["wall_time"] for r in successful),
                    )
                    agg_gbps = (
                        (total_bytes / 1024 / 1024 / 1024) / round_wall
                        if round_wall > 0
                        else 0.0
                    )
                    print(
                        f"  Round result: {agg_gbps:.3f} GB/s aggregate "
                        f"({total_bytes / 1024 / 1024 / 1024:.2f} GB in {round_wall:.2f}s)"
                    )
                else:
                    failed = [r for r in round_results if not r.get("success")]
                    if failed:
                        print(f"  Round FAILED: {failed[0].get('error', 'unknown')}")

            summaries = aggregate_results(all_raw_results)
            print_console_report(summaries)
            export_csv(
                raw_results=all_raw_results,
                summaries=summaries,
                output_dir=args.output_dir,
            )
            if not args.no_charts:
                generate_charts(summaries=summaries, output_dir=args.output_dir)

    # ==========================================================================
    # headers_batch benchmark
    # ==========================================================================
    if benchmark_op in ("headers_batch", "both"):
        hb_combos = generate_headers_batch_combinations(
            backends=args.backends,
            batch_sizes=args.batch_sizes,
            num_threads_list=args.num_threads,
            num_processes_list=args.num_processes,
        )

        if not hb_combos:
            print(
                "WARNING: No valid headers_batch parameter combinations generated. "
                "Make sure at least one non-mock backend is specified."
            )
        else:
            print(f"\n[headers_batch] Total parameter combinations: {len(hb_combos)}")
            # Use a representative buffer_size for reader construction
            buf_bytes = (args.buffer_sizes[0] if args.buffer_sizes else 64) * 1024 * 1024
            all_hb_results: List[Dict[str, Any]] = []

            for idx, combo in enumerate(hb_combos, 1):
                backend = combo["backend"]
                batch_size = combo["batch_size"]
                num_threads = combo["num_threads"]
                nprocs = combo["num_processes"]

                progress = f"[{idx}/{len(hb_combos)}]"
                print(
                    f"\n{'=' * 90}\n"
                    f"{progress} [headers_batch] backend={backend}, "
                    f"batch_size={batch_size}, num_threads={num_threads}, procs={nprocs}\n"
                    f"{'=' * 90}"
                )

                if args.warmup > 0:
                    print(f"  Warmup: {args.warmup} round(s)...")
                    for _w in range(args.warmup):
                        run_headers_batch_round(
                            files=files,
                            mount_point=args.mount_point,
                            backend=backend,
                            num_processes=nprocs,
                            buffer_size=buf_bytes,
                            batch_size=batch_size,
                            num_threads=num_threads,
                            num_iterations=1,
                            verbose=False,
                        )
                    print("  Warmup complete.")

                round_results = run_headers_batch_round(
                    files=files,
                    mount_point=args.mount_point,
                    backend=backend,
                    num_processes=nprocs,
                    buffer_size=buf_bytes,
                    batch_size=batch_size,
                    num_threads=num_threads,
                    num_iterations=args.iterations,
                    verbose=verbose,
                )

                all_hb_results.extend(round_results)

                successful = [r for r in round_results if r.get("success")]
                if successful:
                    total_files = sum(r["total_files_read"] for r in successful)
                    round_wall = successful[0].get(
                        "round_wall_time",
                        max(r["wall_time"] for r in successful),
                    )
                    agg_fps = total_files / round_wall if round_wall > 0 else 0.0
                    print(
                        f"  Round result: {agg_fps:.1f} files/s aggregate "
                        f"({total_files} files in {round_wall:.2f}s)"
                    )
                else:
                    failed = [r for r in round_results if not r.get("success")]
                    if failed:
                        print(f"  Round FAILED: {failed[0].get('error', 'unknown')}")

            hb_summaries = aggregate_headers_batch_results(all_hb_results)
            print_headers_batch_report(hb_summaries)
            export_headers_batch_csv(
                raw_results=all_hb_results,
                summaries=hb_summaries,
                output_dir=args.output_dir,
            )

    benchmark_elapsed = time.time() - benchmark_start
    print(f"\n{'#' * 90}")
    print(f"  Benchmark completed in {benchmark_elapsed:.1f}s")
    print(f"{'#' * 90}\n")

    print(f"\nAll results saved to: {os.path.abspath(args.output_dir)}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
