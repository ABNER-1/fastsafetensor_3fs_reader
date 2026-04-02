# SPDX-License-Identifier: Apache-2.0

"""Backend selection: ``FASTSAFETENSORS_BACKEND`` → cpp / python / mock."""

from __future__ import annotations

import logging
import os
from typing import Any

from .interface import FileReaderInterface
from .mock import MockFileReader

logger = logging.getLogger(__name__)

_VALID_BACKENDS = ("cpp", "python", "mock", "auto")

_BACKEND: str = "mock"
ThreeFSFileReader: type | None = None


def _load_backend(name: str) -> None:
    global ThreeFSFileReader, _BACKEND
    if name == "cpp":
        from .reader_cpp import ThreeFSFileReaderCpp

        ThreeFSFileReader = ThreeFSFileReaderCpp
        _BACKEND = "cpp"
    elif name == "python":
        from .reader_py import ThreeFSFileReaderPy

        ThreeFSFileReader = ThreeFSFileReaderPy
        _BACKEND = "python"
    elif name == "mock":
        ThreeFSFileReader = MockFileReader
        _BACKEND = "mock"
    else:
        raise ValueError(f"Unknown backend: {name!r}")


def init_backend() -> None:
    """Auto-select backend (cpp → python → mock).

    Override with ``FASTSAFETENSORS_BACKEND=cpp|python|mock``.
    """
    forced = os.environ.get("FASTSAFETENSORS_BACKEND", "").lower().strip()
    if forced and forced not in _VALID_BACKENDS:
        raise ValueError(
            f"FASTSAFETENSORS_BACKEND={forced!r} is invalid. "
            f"Valid values: {', '.join(_VALID_BACKENDS)} (or unset)"
        )

    if forced and forced != "auto":
        _load_backend(forced)
        logger.info(
            "using backend=%r (forced via FASTSAFETENSORS_BACKEND)",
            _BACKEND,
        )
    else:
        for candidate in ("cpp", "python"):
            try:
                _load_backend(candidate)
                logger.info(
                    "using backend=%r (auto-selected)", _BACKEND
                )
                break
            except ImportError as exc:
                logger.debug(
                    "backend=%r not available (%s), trying next",
                    candidate,
                    exc,
                )

        if ThreeFSFileReader is None:
            _load_backend("mock")
            logger.warning(
                "no real 3FS backend available "
                "(cpp/python both failed), falling back to mock backend"
            )


def is_available() -> bool:
    return _BACKEND in ("cpp", "python")


def get_backend() -> str:
    return _BACKEND


def create_reader(backend: str = "auto", **kwargs: Any) -> FileReaderInterface:
    """Create a reader instance, optionally forcing a specific backend.

    ``**kwargs`` are forwarded to the reader constructor.
    """
    if backend == "auto":
        if ThreeFSFileReader is None:
            raise RuntimeError("No backend is available")
        return ThreeFSFileReader(**kwargs)
    elif backend == "cpp":
        from .reader_cpp import ThreeFSFileReaderCpp

        return ThreeFSFileReaderCpp(**kwargs)
    elif backend == "python":
        from .reader_py import ThreeFSFileReaderPy

        return ThreeFSFileReaderPy(**kwargs)
    elif backend == "mock":
        return MockFileReader(**kwargs)
    else:
        raise ValueError(
            f"backend={backend!r} is invalid. Valid values: {', '.join(_VALID_BACKENDS)}"
        )


__all__ = [
    "ThreeFSFileReader",
    "init_backend",
    "is_available",
    "get_backend",
    "create_reader",
]
