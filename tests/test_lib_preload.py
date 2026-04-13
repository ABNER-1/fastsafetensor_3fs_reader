# SPDX-License-Identifier: Apache-2.0

"""Tests for the _lib_preload module (libhf3fs_api_shared.so auto-discovery)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

from fastsafetensor_3fs_reader import _lib_preload

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_fake_so(directory: Path, name: str = "libhf3fs_api_shared.so") -> Path:
    """Create a zero-byte file that looks like a shared library."""
    directory.mkdir(parents=True, exist_ok=True)
    so_path = directory / name
    so_path.touch()
    return so_path


def _reset_preload_state():
    """Reset the module-level preload state so each test starts clean."""
    _lib_preload._preloaded_path = None


# ---------------------------------------------------------------------------
# TestFindInEnvVar
# ---------------------------------------------------------------------------


class TestFindInEnvVar:
    """Tests for ``_find_in_env_var()``."""

    def test_returns_none_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("HF3FS_LIB_DIR", raising=False)
        assert _lib_preload._find_in_env_var() is None

    def test_returns_none_when_dir_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF3FS_LIB_DIR", str(tmp_path))
        assert _lib_preload._find_in_env_var() is None

    def test_finds_exact_name(self, tmp_path, monkeypatch):
        so = _create_fake_so(tmp_path)
        monkeypatch.setenv("HF3FS_LIB_DIR", str(tmp_path))
        result = _lib_preload._find_in_env_var()
        assert result == str(so)

    def test_finds_versioned_name(self, tmp_path, monkeypatch):
        _create_fake_so(tmp_path, "libhf3fs_api_shared.so.1.0.0")
        monkeypatch.setenv("HF3FS_LIB_DIR", str(tmp_path))
        result = _lib_preload._find_in_env_var()
        assert result is not None
        assert "libhf3fs_api_shared.so" in result


# ---------------------------------------------------------------------------
# TestFindInLdLibraryPath
# ---------------------------------------------------------------------------


class TestFindInLdLibraryPath:
    """Tests for ``_find_in_ld_library_path()``."""

    def test_returns_none_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        assert _lib_preload._find_in_ld_library_path() is None

    def test_returns_none_when_dirs_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path))
        assert _lib_preload._find_in_ld_library_path() is None

    def test_finds_in_single_dir(self, tmp_path, monkeypatch):
        so = _create_fake_so(tmp_path)
        monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path))
        result = _lib_preload._find_in_ld_library_path()
        assert result == str(so)

    def test_finds_in_second_dir(self, tmp_path, monkeypatch):
        dir1 = tmp_path / "empty"
        dir1.mkdir()
        dir2 = tmp_path / "has_lib"
        so = _create_fake_so(dir2)
        monkeypatch.setenv("LD_LIBRARY_PATH", f"{dir1}{os.pathsep}{dir2}")
        result = _lib_preload._find_in_ld_library_path()
        assert result == str(so)

    def test_finds_versioned_name(self, tmp_path, monkeypatch):
        _create_fake_so(tmp_path, "libhf3fs_api_shared.so.2")
        monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path))
        result = _lib_preload._find_in_ld_library_path()
        assert result is not None
        assert "libhf3fs_api_shared.so" in result


# ---------------------------------------------------------------------------
# TestFindInPipPackages
# ---------------------------------------------------------------------------


class TestFindInPipPackages:
    """Tests for ``_find_in_pip_packages()``."""

    def test_returns_none_when_package_not_installed(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            assert _lib_preload._find_in_pip_packages() is None

    def test_returns_none_when_spec_has_no_origin(self):
        fake_spec = mock.MagicMock()
        fake_spec.origin = None
        with mock.patch("importlib.util.find_spec", return_value=fake_spec):
            assert _lib_preload._find_in_pip_packages() is None

    def test_finds_in_libs_dir(self, tmp_path):
        """Simulates auditwheel .libs/ directory layout."""
        # Create fake package structure:
        #   site-packages/hf3fs_py_usrbio/__init__.py
        #   site-packages/hf3fs_py_usrbio.libs/libhf3fs_api_shared.so
        site_packages = tmp_path / "site-packages"
        pkg_dir = site_packages / "hf3fs_py_usrbio"
        pkg_dir.mkdir(parents=True)
        init_py = pkg_dir / "__init__.py"
        init_py.touch()

        libs_dir = site_packages / "hf3fs_py_usrbio.libs"
        so = _create_fake_so(libs_dir)

        fake_spec = mock.MagicMock()
        fake_spec.origin = str(init_py)

        with mock.patch("importlib.util.find_spec", return_value=fake_spec):
            result = _lib_preload._find_in_pip_packages()
            assert result == str(so)

    def test_finds_in_pkg_lib_subdir(self, tmp_path):
        """Finds the so in <package>/lib/ directory."""
        site_packages = tmp_path / "site-packages"
        pkg_dir = site_packages / "hf3fs_py_usrbio"
        pkg_dir.mkdir(parents=True)
        init_py = pkg_dir / "__init__.py"
        init_py.touch()

        lib_dir = pkg_dir / "lib"
        so = _create_fake_so(lib_dir)

        fake_spec = mock.MagicMock()
        fake_spec.origin = str(init_py)

        with mock.patch("importlib.util.find_spec", return_value=fake_spec):
            result = _lib_preload._find_in_pip_packages()
            assert result == str(so)

    def test_finds_in_pkg_dir_directly(self, tmp_path):
        """Finds the so directly in the package directory."""
        site_packages = tmp_path / "site-packages"
        pkg_dir = site_packages / "hf3fs_py_usrbio"
        pkg_dir.mkdir(parents=True)
        init_py = pkg_dir / "__init__.py"
        init_py.touch()

        so = _create_fake_so(pkg_dir)

        fake_spec = mock.MagicMock()
        fake_spec.origin = str(init_py)

        with mock.patch("importlib.util.find_spec", return_value=fake_spec):
            result = _lib_preload._find_in_pip_packages()
            assert result == str(so)

    def test_returns_none_when_no_so_found(self, tmp_path):
        """Returns None when the package exists but no so is present."""
        site_packages = tmp_path / "site-packages"
        pkg_dir = site_packages / "hf3fs_py_usrbio"
        pkg_dir.mkdir(parents=True)
        init_py = pkg_dir / "__init__.py"
        init_py.touch()

        fake_spec = mock.MagicMock()
        fake_spec.origin = str(init_py)

        with mock.patch("importlib.util.find_spec", return_value=fake_spec):
            assert _lib_preload._find_in_pip_packages() is None

    def test_handles_module_not_found_error(self):
        """Gracefully handles ModuleNotFoundError from find_spec."""
        with mock.patch(
            "importlib.util.find_spec", side_effect=ModuleNotFoundError("no module")
        ):
            assert _lib_preload._find_in_pip_packages() is None


# ---------------------------------------------------------------------------
# TestPreloadHf3fsLibrary
# ---------------------------------------------------------------------------


class TestPreloadHf3fsLibrary:
    """Tests for ``preload_hf3fs_library()``."""

    def setup_method(self):
        _reset_preload_state()

    def teardown_method(self):
        _reset_preload_state()

    def test_returns_false_when_nothing_found(self, monkeypatch):
        monkeypatch.delenv("HF3FS_LIB_DIR", raising=False)
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        with mock.patch.object(_lib_preload, "_find_in_pip_packages", return_value=None):
            assert _lib_preload.preload_hf3fs_library() is False

    def test_does_not_raise_on_failure(self, monkeypatch):
        """preload_hf3fs_library must never raise — it's a best-effort fallback."""
        monkeypatch.delenv("HF3FS_LIB_DIR", raising=False)
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        with mock.patch.object(_lib_preload, "_find_in_pip_packages", return_value=None):
            # Should not raise
            result = _lib_preload.preload_hf3fs_library()
            assert result is False

    def test_env_var_takes_priority(self, tmp_path, monkeypatch):
        """HF3FS_LIB_DIR should be checked before LD_LIBRARY_PATH and pip."""
        env_dir = tmp_path / "env"
        _create_fake_so(env_dir)
        monkeypatch.setenv("HF3FS_LIB_DIR", str(env_dir))

        # Mock ctypes.CDLL to avoid actually loading a fake file
        with mock.patch("ctypes.CDLL") as mock_cdll:
            result = _lib_preload.preload_hf3fs_library()
            assert result is True
            # Verify CDLL was called with the env var path
            call_args = mock_cdll.call_args
            assert str(env_dir) in call_args[0][0]

    def test_idempotent_second_call(self, tmp_path, monkeypatch):
        """Second call returns True immediately without re-loading."""
        env_dir = tmp_path / "env"
        _create_fake_so(env_dir)
        monkeypatch.setenv("HF3FS_LIB_DIR", str(env_dir))

        with mock.patch("ctypes.CDLL") as mock_cdll:
            assert _lib_preload.preload_hf3fs_library() is True
            assert mock_cdll.call_count == 1

            # Second call should not invoke CDLL again
            assert _lib_preload.preload_hf3fs_library() is True
            assert mock_cdll.call_count == 1

    def test_falls_back_to_pip_when_env_not_set(self, tmp_path, monkeypatch):
        """When no env vars are set, falls back to pip package discovery."""
        monkeypatch.delenv("HF3FS_LIB_DIR", raising=False)
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

        pip_so = str(tmp_path / "fake.so")
        Path(pip_so).touch()

        with mock.patch.object(
            _lib_preload, "_find_in_pip_packages", return_value=pip_so
        ):
            with mock.patch("ctypes.CDLL") as mock_cdll:
                result = _lib_preload.preload_hf3fs_library()
                assert result is True
                mock_cdll.assert_called_once_with(pip_so, mode=mock.ANY)

    def test_handles_cdll_oserror(self, tmp_path, monkeypatch):
        """If ctypes.CDLL raises OSError, preload returns False gracefully."""
        env_dir = tmp_path / "env"
        _create_fake_so(env_dir)
        monkeypatch.setenv("HF3FS_LIB_DIR", str(env_dir))

        with mock.patch("ctypes.CDLL", side_effect=OSError("cannot load")):
            result = _lib_preload.preload_hf3fs_library()
            assert result is False


# ---------------------------------------------------------------------------
# TestGetHf3fsLibPath
# ---------------------------------------------------------------------------


class TestGetHf3fsLibPath:
    """Tests for ``get_hf3fs_lib_path()``."""

    def setup_method(self):
        _reset_preload_state()

    def teardown_method(self):
        _reset_preload_state()

    def test_returns_none_before_preload(self):
        assert _lib_preload.get_hf3fs_lib_path() is None

    def test_returns_path_after_successful_preload(self, tmp_path, monkeypatch):
        env_dir = tmp_path / "env"
        so = _create_fake_so(env_dir)
        monkeypatch.setenv("HF3FS_LIB_DIR", str(env_dir))

        with mock.patch("ctypes.CDLL"):
            _lib_preload.preload_hf3fs_library()
            result = _lib_preload.get_hf3fs_lib_path()
            assert result == str(so)

    def test_returns_none_after_failed_preload(self, monkeypatch):
        monkeypatch.delenv("HF3FS_LIB_DIR", raising=False)
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        with mock.patch.object(_lib_preload, "_find_in_pip_packages", return_value=None):
            _lib_preload.preload_hf3fs_library()
            assert _lib_preload.get_hf3fs_lib_path() is None
