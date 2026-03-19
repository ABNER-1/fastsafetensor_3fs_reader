
// SPDX-License-Identifier: Apache-2.0

/**
 * Lightweight 3FS USRBIO C++ Reader (v2)
 *
 * This pybind11 module dynamically links against libhf3fs_api_shared.so
 * instead of embedding the 3FS source files.  GPU memory transfer is
 * handled in C++ via CUDA Runtime API (cudaMemcpy) for GPU targets,
 * or plain memcpy for host targets.
 *
 * Build requirements:
 *   - pybind11
 *   - libhf3fs_api_shared.so (set HF3FS_LIB_DIR)
 *   - CUDA Runtime (cuda_runtime.h + libcudart)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <vector>

#include <sys/stat.h>

// 3FS USRBIO header (bundled from deepseek-ai/3FS for build convenience)
#include "include/hf3fs_usrbio.h"

// CUDA Runtime for GPU memory transfer (cudaMemcpy, cudaPointerGetAttributes)
#include <cuda_runtime.h>

namespace py = pybind11;

// =========================================================================
// Debug helpers
// =========================================================================

namespace {

bool debug_enabled() {
    static const bool val = [] {
        const char *e = std::getenv("FASTSAFETENSORS_DEBUG");
        if (!e) return false;
        std::string s(e);
        for (auto &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s == "true" || s == "1";
    }();
    return val;
}

template <typename... Args>
void dbg(const char *fmt, Args... args) {
    if (debug_enabled()) {
        fprintf(stderr, "[ThreeFSReader_v2] ");
        fprintf(stderr, fmt, args...);
        fprintf(stderr, "\n");
        fflush(stderr);
    }
}



} // anonymous namespace

// =========================================================================
// ThreeFSReader class
// =========================================================================

class ThreeFSReader {
private:
    // --- member variables (declared first for forward reference) -----------
    std::string mount_point_;
    int entries_;
    uint64_t buffer_size_;

    struct hf3fs_iov iov_{};
    struct hf3fs_ior ior_{};
    bool iov_pinned_ = false;  // true if iov_.base is registered as CUDA pinned memory

    std::mutex mu_;
    std::map<std::string, int> fd_map_;

    // --- private helpers ---------------------------------------------------

    void close_all_internal() {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto &kv : fd_map_) {
            hf3fs_dereg_fd(kv.second);
            ::close(kv.second);
        }
        fd_map_.clear();
    }

    void copy_iov_to_target(uintptr_t target, int64_t nbytes) {
        if (target == 0 || nbytes == 0) return;

        // Query whether target is a CUDA device/managed pointer.
        // cudaGetLastError() is called immediately after to clear any sticky
        // error that cudaPointerGetAttributes may set when CUDA context is not
        // yet initialised (common in multi-process spawn scenarios).
        cudaPointerAttributes attrs;
        cudaError_t err = cudaPointerGetAttributes(&attrs,
                                                   reinterpret_cast<void *>(target));
        cudaGetLastError();  // clear sticky error unconditionally

        bool is_device = (err == cudaSuccess &&
                          (attrs.type == cudaMemoryTypeDevice ||
                           attrs.type == cudaMemoryTypeManaged));

        if (is_device) {
            // Confirmed GPU pointer: explicit host-to-device copy.
            cudaError_t cpy = cudaMemcpy(reinterpret_cast<void *>(target),
                                         iov_.base,
                                         static_cast<size_t>(nbytes),
                                         cudaMemcpyHostToDevice);
            if (cpy != cudaSuccess)
                throw std::runtime_error(
                    std::string("cudaMemcpy H2D failed: ") +
                    cudaGetErrorString(cpy));
        } else {
            // cudaPointerGetAttributes failed or returned an unregistered/host
            // type.  Do NOT blindly call memcpy here — if target is actually a
            // GPU address (e.g. CUDA context not yet initialised when the query
            // ran), memcpy would SIGSEGV.
            //
            // Instead, try cudaMemcpyDefault which lets the CUDA runtime infer
            // the memory type of both src and dst at call time.  This correctly
            // handles GPU pointers even when cudaPointerGetAttributes failed.
            cudaError_t cpy = cudaMemcpy(reinterpret_cast<void *>(target),
                                         iov_.base,
                                         static_cast<size_t>(nbytes),
                                         cudaMemcpyDefault);
            if (cpy != cudaSuccess) {
                // cudaMemcpyDefault also failed: CUDA is completely unavailable.
                // Only fall back to host memcpy when we are confident target is
                // a host address (CUDA not present at all).
                dbg("cudaMemcpyDefault failed (err=%d), falling back to memcpy",
                    (int)cpy);
                cudaGetLastError();  // clear error before host fallback
                memcpy(reinterpret_cast<void *>(target), iov_.base,
                       static_cast<size_t>(nbytes));
            }
        }
    }

public:
    ThreeFSReader(const std::string &mount_point,
                  int entries,
                  int io_depth,
                  uint64_t buffer_size)
        : mount_point_(mount_point),
          entries_(entries),
          buffer_size_(buffer_size) {

        dbg("init: mount=%s entries=%d io_depth=%d buf=%lu",
            mount_point.c_str(), entries, io_depth,
            (unsigned long)buffer_size);

        // Create IOV (shared-memory region)
        int ret = hf3fs_iovcreate(&iov_, mount_point.c_str(),
                                  buffer_size, 0, -1);
        if (ret != 0)
            throw std::runtime_error("hf3fs_iovcreate failed: " +
                                     std::to_string(ret));

        // Create IOR (I/O ring)
        ret = hf3fs_iorcreate4(&ior_, mount_point.c_str(), entries,
                               /*for_read=*/true, io_depth,
                               /*timeout=*/0, /*numa=*/-1, /*flags=*/0);
        if (ret != 0) {
            hf3fs_iovdestroy(&iov_);
            throw std::runtime_error("hf3fs_iorcreate4 failed: " +
                                     std::to_string(ret));
        }

        // Register IOV buffer as CUDA pinned memory for faster H2D transfers.
        // cudaHostRegister allows the CUDA DMA engine to transfer directly from
        // this buffer without an extra kernel-space copy.  Failure is silently
        // ignored so the reader still works with ordinary pageable memory.
        cudaError_t pin_err = cudaHostRegister(
            iov_.base, buffer_size,
            cudaHostRegisterDefault);
        if (pin_err == cudaSuccess) {
            iov_pinned_ = true;
            dbg("init OK  iov.base=%p iov.size=%lu (pinned)",
                (void *)iov_.base, (unsigned long)iov_.size);
        } else {
            cudaGetLastError();  // clear sticky error
            dbg("init OK  iov.base=%p iov.size=%lu (not pinned, cuda_err=%d)",
                (void *)iov_.base, (unsigned long)iov_.size, (int)pin_err);
        }
    }

    ~ThreeFSReader() {
        close_all_internal();
        if (ior_.iorh) hf3fs_iordestroy(&ior_);
        // Unregister pinned memory before destroying the IOV.
        // Failure is silently ignored: CUDA context may already be destroyed
        // at process exit, in which case CUDA cleans up automatically.
        if (iov_pinned_ && iov_.base) {
            cudaHostUnregister(iov_.base);
            cudaGetLastError();  // clear any sticky error
        }
        if (iov_.iovh) hf3fs_iovdestroy(&iov_);
    }

    // non-copyable
    ThreeFSReader(const ThreeFSReader &) = delete;
    ThreeFSReader &operator=(const ThreeFSReader &) = delete;

    // ----- file management -------------------------------------------------

    int open_file(const std::string &path, int flags = O_RDONLY) {
        int fd = ::open(path.c_str(), flags, 0644);
        if (fd < 0)
            throw std::runtime_error("open failed: " +
                                     std::string(strerror(errno)));
        int ret = hf3fs_reg_fd(fd, 0);
        if (ret > 0) {
            ::close(fd);
            throw std::runtime_error("hf3fs_reg_fd failed: " +
                                     std::string(strerror(ret)));
        }
        std::lock_guard<std::mutex> lk(mu_);
        fd_map_[path] = fd;
        return fd;
    }

    int get_fd(const std::string &path) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = fd_map_.find(path);
        if (it == fd_map_.end())
            throw std::runtime_error("no fd for path: " + path);
        return it->second;
    }

    void close_all() { close_all_internal(); }

    // ----- chunked read (GIL released) -------------------------------------

    int64_t read_chunked(int fd, uintptr_t dev_ptr,
                         int64_t file_offset, int64_t total_length,
                         int64_t chunk_size) {
        if (chunk_size <= 0)
            chunk_size = std::min(total_length, static_cast<int64_t>(buffer_size_));

        int64_t bytes_done = 0;
        int64_t remaining  = total_length;
        int64_t cur_foff   = file_offset;
        int64_t cur_doff   = 0;

        // --- per-call timing accumulators (active only when debug_enabled()) ---
        // Each phase is accumulated across all chunks; a single summary line is
        // printed after the loop so the log is never spammy.
        const bool do_time = debug_enabled();
        double t_prep_submit = 0.0, t_wait = 0.0, t_copy = 0.0;
        int    chunk_count   = 0;
        auto now_sec = []() -> double {
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            return ts.tv_sec + ts.tv_nsec * 1e-9;
        };
        double t_total_start = do_time ? now_sec() : 0.0;

        while (remaining > 0) {
            int64_t buf_limit = static_cast<int64_t>(buffer_size_);
            int64_t this_chunk = std::min(remaining, std::min(chunk_size, buf_limit));

            // --- phase 1: USRBIO prep + submit ----------------------------
            double t0 = do_time ? now_sec() : 0.0;

            int idx = hf3fs_prep_io(&ior_, &iov_, /*read=*/true,
                                    iov_.base, fd, cur_foff, this_chunk,
                                    nullptr);
            if (idx < 0)
                throw std::runtime_error("hf3fs_prep_io failed: " +
                                         std::to_string(idx));

            int sret = hf3fs_submit_ios(&ior_);
            if (sret < 0)
                throw std::runtime_error("hf3fs_submit_ios failed: " +
                                         std::to_string(sret));

            if (do_time) t_prep_submit += now_sec() - t0;

            // --- phase 2: USRBIO wait (3FS async I/O completion) ----------
            double t1 = do_time ? now_sec() : 0.0;

            struct hf3fs_cqe cqe;
            int wret = hf3fs_wait_for_ios(&ior_, &cqe, 1, 1, nullptr);
            if (wret < 0)
                throw std::runtime_error("hf3fs_wait_for_ios failed: " +
                                         std::to_string(wret));

            if (do_time) t_wait += now_sec() - t1;

            int64_t actual = cqe.result;
            if (actual < 0)
                throw std::runtime_error("I/O error: " +
                                         std::to_string(actual));
            if (actual == 0) break; // EOF

            // --- phase 3: copy IOV -> target (GPU or host) ----------------
            // Guard against dev_ptr==0 (download-only mode): after the first
            // chunk cur_doff > 0, so dev_ptr + cur_doff would be a non-zero
            // but invalid address, causing memcpy to SIGSEGV.
            double t2 = do_time ? now_sec() : 0.0;

            if (dev_ptr != 0) {
                copy_iov_to_target(dev_ptr + cur_doff, actual);
            }

            if (do_time) t_copy += now_sec() - t2;

            bytes_done += actual;
            cur_foff   += actual;
            cur_doff   += actual;
            remaining  -= actual;
            ++chunk_count;

            if (actual < this_chunk) break; // short read -> EOF
        }

        // --- one-shot summary log (printed once per read_chunked call) ----
        if (do_time) {
            double t_total = now_sec() - t_total_start;
            dbg("read_chunked fd=%d total=%lld bytes chunks=%d"
                " | prep+submit=%.2fms wait=%.2fms copy=%.2fms total=%.2fms",
                fd, (long long)bytes_done, chunk_count,
                t_prep_submit * 1e3, t_wait * 1e3,
                t_copy * 1e3, t_total * 1e3);
        }

        return bytes_done;
    }

    // ----- batch header read (GIL released) --------------------------------

    using HeaderResult = std::tuple<std::string, std::string, int64_t, int64_t>;

    std::vector<HeaderResult>
    open_and_read_headers(const std::vector<std::string> &paths,
                          int num_threads) {
        std::vector<HeaderResult> results(paths.size());
        std::vector<std::thread> threads;

        auto worker = [this, &paths, &results](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                const auto &p = paths[i];
                int fd = ::open(p.c_str(), O_RDONLY);
                if (fd < 0) {
                    throw std::runtime_error("open failed: " + p + ": " +
                                             strerror(errno));
                }

                struct stat st;
                if (fstat(fd, &st) < 0) {
                    ::close(fd);
                    throw std::runtime_error("fstat failed: " + p);
                }
                int64_t file_size = st.st_size;

                // Read 8-byte header length
                uint8_t hdr_buf[8];
                ssize_t n = pread(fd, hdr_buf, 8, 0);
                if (n < 8) {
                    ::close(fd);
                    throw std::runtime_error("header too short: " + p);
                }
                uint64_t hdr_size = 0;
                memcpy(&hdr_size, hdr_buf, 8); // little-endian

                // Read JSON header
                std::string json(hdr_size, '\0');
                n = pread(fd, json.data(), hdr_size, 8);
                if (n < static_cast<ssize_t>(hdr_size)) {
                    ::close(fd);
                    throw std::runtime_error("incomplete header: " + p);
                }

                // Register fd for later USRBIO use
                int rr = hf3fs_reg_fd(fd, 0);
                if (rr > 0) {
                    ::close(fd);
                    throw std::runtime_error("hf3fs_reg_fd failed: " + p);
                }

                {
                    std::lock_guard<std::mutex> lk(mu_);
                    fd_map_[p] = fd;
                }

                results[i] = HeaderResult{p, std::move(json),
                                          static_cast<int64_t>(hdr_size + 8),
                                          file_size};
            }
        };

        // Partition work across threads
        size_t total = paths.size();
        int nt = std::min(static_cast<int>(total), std::max(1, num_threads));
        size_t per = (total + nt - 1) / nt;

        for (int t = 0; t < nt; ++t) {
            size_t b = t * per;
            size_t e = std::min(b + per, total);
            if (b >= total) break;
            threads.emplace_back(worker, b, e);
        }
        for (auto &th : threads) th.join();

        return results;
    }

    // ----- properties ------------------------------------------------------

    uintptr_t   get_iov_base()    const { return reinterpret_cast<uintptr_t>(iov_.base); }
    uint64_t    get_iov_length()  const { return iov_.size; }
    std::string get_mount_point() const { return mount_point_; }

    static bool check_library() {
        // If we got here the .so loaded successfully.
        return true;
    }
}; // end class ThreeFSReader

// =========================================================================
// pybind11 module definition (must be at file/namespace scope, NOT inside a class)
// =========================================================================

PYBIND11_MODULE(_core_v2, m) {
    m.doc() = "Lightweight 3FS USRBIO reader (v2) - links libhf3fs_api_shared.so";

    m.def("check_library", &ThreeFSReader::check_library,
          "Check if 3FS USRBIO library is available");

    py::class_<ThreeFSReader>(m, "ThreeFSReader")
        .def(py::init<const std::string &, int, int, uint64_t>(),
             py::arg("mount_point"),
             py::arg("entries") = 64,
             py::arg("io_depth") = 0,
             py::arg("buffer_size") = 1073741824ULL)

        .def("open", &ThreeFSReader::open_file,
             py::arg("path"), py::arg("flags") = O_RDONLY)

        .def("get_fd", &ThreeFSReader::get_fd,
             py::arg("path"))

        .def("close_all", &ThreeFSReader::close_all)

        .def("read_chunked", &ThreeFSReader::read_chunked,
             py::arg("fd"), py::arg("dev_ptr"),
             py::arg("file_offset"), py::arg("total_length"),
             py::arg("chunk_size") = 0,
             py::call_guard<py::gil_scoped_release>())

        .def("open_and_read_headers",
             &ThreeFSReader::open_and_read_headers,
             py::arg("paths"), py::arg("num_threads") = 8,
             py::call_guard<py::gil_scoped_release>())

        .def_property_readonly("iov_base", &ThreeFSReader::get_iov_base)
        .def_property_readonly("iov_length", &ThreeFSReader::get_iov_length)
        .def_property_readonly("mount_point", &ThreeFSReader::get_mount_point);
}
