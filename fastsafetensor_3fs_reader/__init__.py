# SPDX-License-Identifier: Apache-2.0

from .interface import FileReaderInterface
from .mock import MockFileReader

_AVAILABLE = False
try:
    from .reader import ThreeFSFileReader, Iov, new_threefs_file_reader
    _AVAILABLE = True
except ImportError:
    ThreeFSFileReader = None
    Iov = None
    new_threefs_file_reader = None

def is_available() -> bool:
    """Check if 3FS C++ module is available."""
    return _AVAILABLE

__all__ = [
    "FileReaderInterface",
    "ThreeFSFileReader",
    "MockFileReader",
    "Iov",
    "new_threefs_file_reader",
    "is_available",
]
