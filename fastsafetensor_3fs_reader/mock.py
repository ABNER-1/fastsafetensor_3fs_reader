# SPDX-License-Identifier: Apache-2.0

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from .interface import FileReaderInterface

class MockFileReader(FileReaderInterface):
    """Local filesystem-backed mock for CI tests. No 3FS/CUDA/C++ required."""

    def __init__(self, mount_point: str = ""):
        self._fd_map: Dict[str, int] = {}
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
            ctypes.memmove(dev_ptr, data, len(data))

        return len(data)

    def read_headers_batch(
        self,
        paths: List[str],
        num_threads: int = 8,
    ) -> Dict[str, Tuple[str, int, int]]:
        if not paths:
            return {}

        results: Dict[str, Tuple[str, int, int]] = {}
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = {pool.submit(self._read_single_header, p): p for p in paths}
            for future in as_completed(futures):
                path = futures[future]
                results[path] = future.result()
        return results

    def _read_single_header(self, path: str) -> Tuple[str, int, int]:
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
