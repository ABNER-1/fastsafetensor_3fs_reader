#!/usr/bin/env python3
"""Simplified USRBIO concurrent read benchmark.

Usage:
    python test_usrbio_simple.py /mnt/3fs 4
"""

import glob
import logging
import multiprocessing as mp
import sys

from test_usrbio_common import print_test_results, run_concurrent_test

# Configure logging to output to stdout by default
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python test_usrbio_simple.py <mount_point> <num_processes> [file_pattern] [iterations] [--download-only]"
        )
        print(
            "Example: python test_usrbio_simple.py /mnt/3fs 4 '/mnt/3fs/model-*.safetensors' 3"
        )
        print(
            "         python test_usrbio_simple.py /mnt/3fs 8 '/mnt/3fs/*.safetensors' 1 --download-only"
        )
        sys.exit(1)

    mount_point = sys.argv[1]
    num_processes = int(sys.argv[2])
    file_pattern = sys.argv[3] if len(sys.argv) > 3 else f"{mount_point}/*.safetensors"
    num_iterations = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    download_only = "--download-only" in sys.argv

    files = sorted(glob.glob(file_pattern))

    if not files:
        print(f"No files found: {file_pattern}")
        sys.exit(1)

    print(f"Found {len(files)} files matching pattern")

    buffer_size = 1024 * 1024 * 1024  # 1GB
    chunk_size = 64 * 1024 * 1024  # 64MB

    results = run_concurrent_test(
        files=files,
        mount_point=mount_point,
        num_processes=num_processes,
        buffer_size=buffer_size,
        chunk_size=chunk_size,
        num_iterations=num_iterations,
        verbose=True,
        download_only=download_only,
    )

    print_test_results(
        results=results,
        num_iterations=num_iterations,
        total_files=len(files),
        verbose=True,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
