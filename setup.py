# SPDX-License-Identifier: Apache-2.0

"""
Build configuration for fastsafetensor_3fs_reader.

Compiles the C++ pybind11 module (_core) that wraps 3FS USRBIO API.
Links directly against UsrbIo.cc and Shm.cc to eliminate runtime
dependency on libhf3fs_api_shared.so.
"""

import os
import shutil

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def find_cuda_home():
    """Find CUDA installation path."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        nvcc_path = shutil.which("nvcc")
        if nvcc_path:
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
    return cuda_home


class OptionalBuildExt(build_ext):
    """Custom build_ext that treats the C++ extension as optional."""

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except Exception as exc:
            print(f"Warning: Failed to build optional extension {ext.name}")
            print(f"  Error: {exc}")
            print(f"  Skipping {ext.name} (this is optional)")


def build_core_extension():
    """Build the _core pybind11 extension module."""
    import pybind11

    pybind11_include = os.path.join(os.path.dirname(pybind11.__file__), "include")
    cpp_dir = os.path.join("fastsafetensor_3fs_reader", "cpp")

    # Source files: pybind11 wrapper + 3FS API implementation
    sources = [
        os.path.join(cpp_dir, "usrbio_reader.cpp"),
        os.path.join(cpp_dir, "3fs_client", "api", "UsrbIo.cc"),
        os.path.join(cpp_dir, "3fs_client", "common", "Shm.cc"),
    ]

    include_dirs = [
        pybind11_include,
        cpp_dir,
        os.path.join(cpp_dir, "3fs_client", "api"),
    ]

    # Check for custom 3FS include path
    if "HF3FS_INCLUDE_DIR" in os.environ:
        include_dirs.insert(0, os.environ["HF3FS_INCLUDE_DIR"])

    libraries = ["stdc++", "pthread"]
    library_dirs = []
    extra_compile_args = ["-fvisibility=hidden", "-std=c++17"]
    extra_link_args = []

    # CUDA configuration (needed for cudaMemcpy in usrbio_reader.cpp)
    cuda_home = find_cuda_home()
    if cuda_home and os.path.exists(cuda_home):
        include_dirs.append(os.path.join(cuda_home, "include"))
        cuda_lib_dirs = [
            os.path.join(cuda_home, "lib64"),
            os.path.join(cuda_home, "lib"),
        ]
        library_dirs.extend([d for d in cuda_lib_dirs if os.path.exists(d)])
        libraries.append("cudart")

    return Extension(
        name="fastsafetensor_3fs_reader.cpp._core",
        sources=sources,
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
        optional=True,
    )


# Environment variable to skip C++ extension build (for pure Python wheel)
# Usage: FST3FS_NO_EXT=1 python setup.py bdist_wheel --python-tag=py3
_skip_ext = os.environ.get("FST3FS_NO_EXT", "0") == "1"

setup(
    packages=[
        "fastsafetensor_3fs_reader",
        "fastsafetensor_3fs_reader.cpp",
    ],
    include_package_data=True,
    package_data={
        "fastsafetensor_3fs_reader.cpp": [
            "*.hpp",
            "*.h",
            "3fs_client/api/*.h",
            "3fs_client/api/*.cc",
            "3fs_client/common/*.h",
            "3fs_client/common/*.cc",
        ],
    },
    ext_modules=[] if _skip_ext else [build_core_extension()],
    cmdclass={"build_ext": OptionalBuildExt},
)
