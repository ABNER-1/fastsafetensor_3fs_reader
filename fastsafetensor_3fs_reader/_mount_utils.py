# SPDX-License-Identifier: Apache-2.0
"""Utilities for resolving the 3FS mount point from file paths."""

from __future__ import annotations

try:
    from hf3fs_py_usrbio import extract_mount_point

except ImportError:
    # hf3fs_py_usrbio is not installed (mock / pure-Python fallback path).
    # Provide a no-op stub so that callers can import extract_mount_point
    # unconditionally without try/except.
    def extract_mount_point(path: str) -> str:  # type: ignore[misc]
        """Stub: hf3fs_py_usrbio is not installed; always returns ``""``."""
        return ""


__all__ = ["extract_mount_point"]
