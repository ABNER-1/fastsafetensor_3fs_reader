# fastsafetensor-3fs-reader

3FS USRBIO file reader for fastsafetensors.

This package provides a high-performance reader for 3FS USRBIO files with
two backend implementations (C++ and pure-Python) and a mock for testing.

## Backends

| Backend | Module | Requirements | Performance |
|---------|--------|-------------|-------------|
| **C++** | `reader_cpp.py` | `libhf3fs_api_shared.so` + libtorch + CUDA | Best (GIL-free, native USRBIO async I/O) |
| **Python** | `reader_py.py` | `hf3fs_py_usrbio` (+ optional PyTorch for GPU) | Good (USRBIO via Client API or OS pread) |
| **Mock** | `mock.py` | None | For testing only |

The package auto-selects the best available backend at import time:
C++ → Python → Mock.  Use `get_backend()` to check which one is active.

> **Note:** The C++ backend supports **pipelined mode** (double-buffered async
> H2D copy via `cudaMemcpyAsync`) which overlaps network I/O with GPU memory
> transfer for significantly better throughput.  Pass `pipelined=True` to
> `read_chunked()` to enable it.  The Python backend does not support
> pipelining and will silently fall back to non-pipelined mode.

## Installation

### Pure-Python mode (no C++ compilation)

```bash
FST3FS_NO_EXT=1 pip install .
```

### With C++ extension

Requires `libhf3fs_api_shared.so` (from a 3FS build) and CUDA Runtime.
The `hf3fs_usrbio.h` header is bundled in the package, so no external
header dependency is needed:

```bash
export HF3FS_LIB_DIR=/path/to/3FS/build/lib         # directory with libhf3fs_api_shared.so
pip install .
```

### Automatic `libhf3fs_api_shared.so` discovery

At import time, the package automatically searches for
`libhf3fs_api_shared.so` using the following priority:

1. **`HF3FS_LIB_DIR`** environment variable (user-explicit, highest priority).
2. **`LD_LIBRARY_PATH`** directories (user already configured).
3. **`hf3fs_py_usrbio` pip install path** — if `hf3fs_py_usrbio` is installed
   via pip, the library is typically located in a sibling `.libs/` directory
   (e.g. `site-packages/hf3fs_py_usrbio.libs/`).  This is discovered
   automatically so you don't need to set `LD_LIBRARY_PATH` manually.

The library is pre-loaded with `RTLD_GLOBAL` so that both the C++ and
Python backends can resolve its symbols.  Use `get_hf3fs_lib_path()` to
check which path was loaded:

```python
from fastsafetensor_3fs_reader import get_hf3fs_lib_path
print(get_hf3fs_lib_path())  # e.g. "/path/to/site-packages/hf3fs_py_usrbio.libs/libhf3fs_api_shared.so"
```

### Installing hf3fs_py_usrbio (for the Python backend)

`hf3fs_py_usrbio` is **not** available on PyPI.  It must be built from the
[DeepSeek 3FS](https://github.com/deepseek-ai/3FS) source tree:

```bash
git clone https://github.com/deepseek-ai/3FS
cd 3FS
git submodule update --init --recursive
# Follow 3FS build instructions (cmake, etc.)
# After build, install the Python package:
cd build && pip install ..
```

> **Important:** The default pip-installed `hf3fs_py_usrbio` package is
> suitable for **testing and validation** but is **not recommended for
> production use**.  For production deployments, build 3FS from source with
> optimized compiler flags tailored to your hardware.  Refer to projects like
> [SGLang](https://github.com/sgl-project/sglang) for examples of
> production-grade 3FS compilation workflows.

## Usage

```python
from fastsafetensor_3fs_reader import (
    ThreeFSFileReader,
    MockFileReader,
    is_available,
    get_backend,
)

# Check which backend is active
print(f"Backend: {get_backend()}")  # "cpp", "python", or "mock"

# Use mock reader for testing (always available)
reader = MockFileReader()
headers = reader.read_headers_batch(["/path/to/file.safetensors"])
reader.close()

# Use 3FS reader when available
if is_available():
    reader = ThreeFSFileReader(mount_point="/mnt/3fs")
    headers = reader.read_headers_batch([
        "/mnt/3fs/model-00001.safetensors",
        "/mnt/3fs/model-00002.safetensors",
    ])

    # Read tensor data into GPU memory
    import torch
    buf = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    bytes_read = reader.read_chunked(
        path="/mnt/3fs/model-00001.safetensors",
        dev_ptr=buf.data_ptr(),
        file_offset=0,
        total_length=1024 * 1024,
    )
    reader.close()
```

## Benchmark

The `hack/benchmark/` directory contains a comprehensive benchmarking suite.
Use `benchmark_runner.py` to measure read throughput across different backends,
buffer sizes, chunk sizes, and process counts.

### Full benchmark (read + GPU copy)

```bash
python hack/benchmark/benchmark_runner.py \
    --mount-point /mnt/3fs \
    --backends cpp,python \
    --buffer-sizes 8,16,32,64,128,256,512 \
    --chunk-sizes 8,16,32,64,128,256,512 \
    --num-processes 1,2,4,8 \
    --iterations 3
```

### Download-only benchmark (host memory only, no GPU copy)

```bash
python hack/benchmark/benchmark_runner.py \
    --mount-point /mnt/3fs \
    --backends cpp,python \
    --buffer-sizes 8,16,32,64,128,256,512 \
    --chunk-sizes 8,16,32,64,128,256,512 \
    --num-processes 1,2,4,8 \
    --download-only \
    --iterations 3
```

### Key parameters

| Parameter | Description | Default                       |
|-----------|-------------|-------------------------------|
| `--mount-point` | 3FS FUSE mount-point path | *(required)*                  |
| `--backends` | Comma-separated backend names | `mock,python,cpp`             |
| `--buffer-sizes` | Buffer sizes in MB | `8,16,32,64,128,256,512,1024` |
| `--chunk-sizes` | Chunk sizes in MB | `8,16,32,64,128,256,512,1024`         |
| `--num-processes` | Process counts | `1,2,4,8`                     |
| `--download-only` | Read into host memory only (skip GPU copy) | `false`                       |
| `--iterations` | Iterations per combination | `3`                           |
| `--mode` | `grid` (sweep all combos) or `single` | `grid`                        |
| `--output-dir` | Directory for CSV and chart output | `./benchmark_results`         |

### Performance Results

<!-- TODO: Add benchmark results here -->

> Results will be updated with real benchmark data.

| Configuration | Throughput (GB/s) | Latency (ms) | Notes |
|---------------|-------------------|--------------|-------|
| TBD           | TBD               | TBD          | TBD   |

## License

Apache-2.0
