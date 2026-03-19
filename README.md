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

## Installation

### Pure-Python mode (no C++ compilation)

```bash
FST3FS_NO_EXT=1 pip install .
```

### With C++ extension

Requires `libhf3fs_api_shared.so` (from a 3FS build) and PyTorch.
The `hf3fs_usrbio.h` header is bundled in the package, so no external
header dependency is needed:

```bash
export HF3FS_LIB_DIR=/path/to/3FS/build/lib         # directory with libhf3fs_api_shared.so
pip install .
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

## License

Apache-2.0
