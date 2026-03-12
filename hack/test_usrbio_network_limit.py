#!/usr/bin/env python3
"""Test USRBIO network bandwidth limit with concurrent multi-process reads.

Usage:
    python test_usrbio_network_limit.py --mount-point /mnt/3fs --files /mnt/3fs/model-*.safetensors --num-processes 8
"""

import sys
import time
import argparse
import glob
import multiprocessing as mp

from test_usrbio_common import (
    run_concurrent_test,
    print_test_results,
    print_summary_statistics
)

def main():
    parser = argparse.ArgumentParser(
        description='Test USRBIO network bandwidth limit with concurrent reads',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 8 processes
  python test_usrbio_network_limit.py --mount-point /mnt/3fs --files "/mnt/3fs/model-*.safetensors" --num-processes 8
  
  # Test with custom buffer and chunk sizes
  python test_usrbio_network_limit.py --mount-point /mnt/3fs --files "/mnt/3fs/*.safetensors" --num-processes 4 --buffer-size 2048 --chunk-size 128
  
  # Stress test with many processes
  python test_usrbio_network_limit.py --mount-point /mnt/3fs --files "/mnt/3fs/*.safetensors" --num-processes 16
        """
    )
    
    parser.add_argument('--mount-point', type=str, required=True,
                       help='3FS mount point (e.g., /mnt/3fs)')
    parser.add_argument('--files', type=str, required=True,
                       help='File pattern to read (supports wildcards, e.g., "/mnt/3fs/model-*.safetensors")')
    parser.add_argument('--num-processes', type=int, default=4,
                       help='Number of concurrent processes (default: 4)')
    parser.add_argument('--buffer-size', type=int, default=1024,
                       help='USRBIO buffer size in MB (default: 1024)')
    parser.add_argument('--chunk-size', type=int, default=64,
                       help='Read chunk size in MB (default: 64)')
    parser.add_argument('--iterations', type=int, default=1,
                       help='Number of times each process reads its assigned files (default: 1)')
    parser.add_argument('--repeat', type=int, default=1,
                       help='Number of times to repeat the entire test (default: 1)')
    parser.add_argument('--download-only', action='store_true',
                       help='Test pure download performance (no GPU copy)')
    
    args = parser.parse_args()
    
    files = sorted(glob.glob(args.files))
    
    if not files:
        print(f"Error: No files found matching pattern: {args.files}")
        sys.exit(1)
    
    print(f"Found {len(files)} files matching pattern")
    
    buffer_size = args.buffer_size * 1024 * 1024  # MB to bytes
    chunk_size = args.chunk_size * 1024 * 1024    # MB to bytes
    
    all_results = []
    for i in range(args.repeat):
        if args.repeat > 1:
            print(f"\n{'#'*80}")
            print(f"# Test iteration {i+1}/{args.repeat}")
            print(f"{'#'*80}\n")
        
        results = run_concurrent_test(
            files=files,
            mount_point=args.mount_point,
            num_processes=args.num_processes,
            buffer_size=buffer_size,
            chunk_size=chunk_size,
            num_iterations=args.iterations,
            verbose=True,
            download_only=args.download_only
        )
        
        print_test_results(
            results=results,
            num_iterations=args.iterations,
            total_files=len(files),
            verbose=True
        )
        
        all_results.append(results)
        
        if i < args.repeat - 1:
            print("Waiting 5 seconds before next iteration...")
            time.sleep(5)
    
    print_summary_statistics(all_results, args.repeat)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
