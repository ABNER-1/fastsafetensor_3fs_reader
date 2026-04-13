# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod


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
        pipelined: bool = False,
    ) -> int:
        """Manages fd automatically: reuses cached fd from read_headers_batch,
        or opens a new one and keeps it for subsequent calls.

        Args:
            pipelined: If True, use double-buffered pipelined I/O with async
                H2D copy to overlap network I/O and GPU transfer.
                Note: only takes effect when dev_ptr points to GPU memory
                and a CUDA stream was successfully created during
                initialization. Falls back to non-pipelined mode silently
                otherwise.

        Returns bytes actually read.
        """
        ...

    @abstractmethod
    def read_headers_batch(
        self,
        paths: list[str],
        num_threads: int = 8,
    ) -> dict[str, tuple[str, int, int]]:
        """Opens each file, caches the fd for later read_chunked reuse.

        Returns Dict[path, (header_json, header_length, file_size)]
        where header_length = 8 + len(header_json_bytes).
        """
        ...

    @abstractmethod
    def close(self) -> None:
        ...
