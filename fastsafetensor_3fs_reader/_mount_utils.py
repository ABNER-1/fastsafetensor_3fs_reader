# SPDX-License-Identifier: Apache-2.0
"""Utilities for resolving the 3FS mount point from file paths."""

from __future__ import annotations

try:
    from hf3fs_py_usrbio import extract_mount_point

except ImportError:
    # No-op stub when hf3fs_py_usrbio is not installed.
    def extract_mount_point(path: str) -> str:  # type: ignore[misc]
        return ""


__all__ = ["extract_mount_point"]
