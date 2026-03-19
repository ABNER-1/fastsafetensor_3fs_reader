# SPDX-License-Identifier: Apache-2.0

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from ._cuda_utils import _copy_host_to_target
from .interface import FileReaderInterface


class MockFileReader(FileReaderInterface):
    """Local filesystem-backed mock for CI tests. No 3FS/CUDA/C++ required."""

    def __init__(self, mount_point: str = "", **kwargs) -> None:
        # Accept and ignore extra keyword arguments (entries, io_depth,
        # buffer_size, etc.) so that MockFileReader can be used as a
        # drop-in replacement for ThreeFSFileReaderCpp / Py without
        # the caller having to strip backend-specific parameters.
        self._fd_map: dict[str, int] = {}
        self._mount_point = mount_point

    def read_chunked(
        self,
        path: str,
        dev_ptr: int,
        file_offset: int,
        total_length: int,
        chunk_size: int = 0,
    ) -> int:
        if path not in self._fd_map:
            self._fd_map[path] = os.open(path, os.O_RDONLY)
        fd = self._fd_map[path]

        data = os.pread(fd, total_length, file_offset)

        if dev_ptr != 0:
            # Use _copy_host_to_target instead of ctypes.memmove so that GPU
            # pointers (cudaMemcpyDefault) are handled safely without SIGSEGV.
            staging_buf = bytearray(data)
            staging_ptr = ctypes.addressof(
                (ctypes.c_char * len(staging_buf)).from_buffer(staging_buf)
            )
            _copy_host_to_target(staging_buf, staging_ptr, dev_ptr, len(data))

        return len(data)

    def read_headers_batch(
        self,
        paths: list[str],
        num_threads: int = 8,
    ) -> dict[str, tuple[str, int, int]]:
        if not paths:
            return {}

        results: dict[str, tuple[str, int, int]] = {}
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = {pool.submit(self._read_single_header, p): p for p in paths}
            for future in as_completed(futures):
                path = futures[future]
                results[path] = future.result()
        return results

    def _read_single_header(self, path: str) -> tuple[str, int, int]:
        fd = os.open(path, os.O_RDONLY)
        self._fd_map[path] = fd
        file_size = os.fstat(fd).st_size

        header_len_bytes = os.pread(fd, 8, 0)
        header_size = int.from_bytes(header_len_bytes, byteorder="little", signed=False)
        header_json = os.pread(fd, header_size, 8).decode("utf-8")

        return (header_json, header_size + 8, file_size)

    def close(self) -> None:
        for fd in self._fd_map.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fd_map.clear()
