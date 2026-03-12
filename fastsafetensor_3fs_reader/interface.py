# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

class FileReaderInterface(ABC):
    """Minimal abstract interface for 3FS file readers.

    fd lifecycle is managed internally; callers only deal with file paths.
    """

    @abstractmethod
    def read_chunked(
        self,
        path: str,
        dev_ptr: int,
        file_offset: int,
        total_length: int,
        chunk_size: int = 0,
    ) -> int:
        """Read file data into target memory address.

        Manages fd automatically: reuses cached fd from read_headers_batch,
        or opens a new one and keeps it for subsequent calls.

        Returns bytes actually read.
        """
        ...

    @abstractmethod
    def read_headers_batch(
        self,
        paths: List[str],
        num_threads: int = 8,
    ) -> Dict[str, Tuple[str, int, int]]:
        """Read SafeTensors headers from multiple files in parallel.

        Opens each file, caches the fd for later read_chunked reuse.

        Returns Dict[path, (header_json, header_length, file_size)]
        where header_length = 8 + len(header_json_bytes).
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close all resources including cached fds."""
        ...
