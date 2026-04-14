# SPDX-License-Identifier: Apache-2.0

"""
Build configuration for fastsafetensor_3fs_reader.

Compiles the C++ pybind11 module (_core_v2) that wraps 3FS USRBIO API.
Dynamically links against libhf3fs_api_shared.so and libcudart at runtime.
GPU memory transfer (host to device) is performed in C++ via cudaMemcpy.
The hf3fs_usrbio.h header is bundled in cpp/include/ so no external
header dependency is required at build time.

Environment variables:
    HF3FS_LIB_DIR     – directory containing libhf3fs_api_shared.so
    FST3FS_NO_EXT=1   – skip C++ extension build entirely
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


def build_core_v2_extension():
    """Build the _core_v2 pybind11 extension module.

    Links dynamically against:
      - libhf3fs_api_shared.so  (3FS USRBIO API)
      - libcudart  (CUDA Runtime, for GPU memory transfer)

    No libtorch/PyTorch C++ dependency â GPU memory transfer is handled
    directly in C++ via CUDA Runtime API.
    """
    import pybind11

    pybind11_include = os.path.join(os.path.dirname(pybind11.__file__), "include")
    cpp_dir = os.path.join("fastsafetensor_3fs_reader", "cpp")

    # Source files: only the v2 wrapper (no embedded 3FS sources)
    sources = [
        os.path.join(cpp_dir, "usrbio_reader_v2.cpp"),
    ]

    # The cpp/ directory is added so that `#include "include/hf3fs_usrbio.h"`
    # resolves to the bundled header.
    include_dirs = [pybind11_include, cpp_dir]
    libraries = ["stdc++", "pthread", "hf3fs_api_shared"]
    library_dirs = []
    extra_compile_args = ["-fvisibility=hidden", "-std=c++17"]
    extra_link_args = []

    # 3FS library directory
    hf3fs_lib_dir = os.environ.get("HF3FS_LIB_DIR")
    if hf3fs_lib_dir:
        library_dirs.append(hf3fs_lib_dir)
        extra_link_args.append(f"-Wl,-rpath,{hf3fs_lib_dir}")

    # CUDA configuration (for cuda_runtime.h + libcudart)
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
        name="fastsafetensor_3fs_reader.cpp._core_v2",
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
            "include/*.h",
        ],
    },
    ext_modules=[] if _skip_ext else [build_core_v2_extension()],
    cmdclass={"build_ext": OptionalBuildExt},
)
