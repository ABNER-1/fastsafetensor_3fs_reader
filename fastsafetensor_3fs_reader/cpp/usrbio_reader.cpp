// SPDX-License-Identifier: Apache-2.0

/**
 * 3FS USRBIO C++ Reader Implementation
 * 
 * High-performance async file reader using DeepSeek AI's 3FS USRBIO API.
 * This implementation avoids GIL contention by releasing GIL during I/O operations.
 * 
 * Key Features:
 * - Async I/O with submit/wait pattern
 * - GIL release during blocking operations
 * - Thread-safe request tracking
 * - Integration with fastsafetensors gds_device_buffer
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <map>
#include <vector>
#include <atomic>
#include <memory>
#include <functional>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>

// CUDA headers for GPU memory operations
#include <cuda_runtime.h>

// For fstat
#include <sys/stat.h>

// BS::thread_pool for parallel file operations
#include "BS_thread_pool.hpp"

// 3FS USRBIO headers (from cloned 3fs_repo)
#include <hf3fs_usrbio.h>

namespace py = pybind11;

// =============================================================================
// Debug Logging Utilities
// =============================================================================

#include <chrono>

namespace {
    bool is_debug_enabled() {
        static const bool enabled = []() {
            const char* env = std::getenv("FASTSAFETENSORS_DEBUG");
            if (env == nullptr) return false;
            std::string val(env);
            // Convert to lowercase
            for (auto& ch : val) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            return val == "true";
        }();
        return enabled;
    }

    template<typename... Args>
    void debug_log(const char* format, Args... args) {
        if (is_debug_enabled()) {
            fprintf(stderr, "[ThreeFSReader DEBUG] ");
            fprintf(stderr, format, args...);
            fprintf(stderr, "\n");
            fflush(stderr);
        }
    }

    void debug_log(const char* message) {
        if (is_debug_enabled()) {
            fprintf(stderr, "[ThreeFSReader DEBUG] %s\n", message);
            fflush(stderr);
        }
    }
    
    // Performance timing utilities
    using TimePoint = std::chrono::high_resolution_clock::time_point;
    
    inline TimePoint now() {
        return std::chrono::high_resolution_clock::now();
    }
    
    inline double elapsed_ms(TimePoint start, TimePoint end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    void perf_log(const char* stage, double ms) {
        if (is_debug_enabled()) {
            fprintf(stderr, "[ThreeFSReader PERF] %s: %.3f ms\n", stage, ms);
            fflush(stderr);
        }
    }
}

// =============================================================================
// C++ Request Tracker
// =============================================================================

struct RequestContext {
    int64_t offset;
    int64_t length;
    int64_t ptr_offset;
    bool completed = false;
    int64_t result = 0;
    
    RequestContext(int64_t o, int64_t l, int64_t p) 
        : offset(o), length(l), ptr_offset(p) {}
};

// =============================================================================
// ThreeFSReader Class
// =============================================================================

class ThreeFSReader {
public:
    /**
     * Construct a 3FS USRBIO reader
     * 
     * @param mount_point: 3FS mount point path
     * @param entries: Maximum concurrent I/O requests
     * @param io_depth: I/O depth control (0 = no limit, >0 = batch threshold, <0 = wait threshold)
     * @param buffer_size: Iov buffer size in bytes
     */
    ThreeFSReader(
        const std::string& mount_point,
        int entries,
        int io_depth,
        uint64_t buffer_size
    )
        : mount_point_(mount_point)
        , entries_(entries)
        , io_depth_(io_depth)
        , buffer_size_(buffer_size)
        , next_req_id_(0)
    {
        debug_log("Initializing ThreeFSReader: mount_point=%s, entries=%d, io_depth=%d, buffer_size=%lu",
                  mount_point.c_str(), entries, io_depth, buffer_size);
        
        // Create Iov (shared memory region)
        int ret = hf3fs_iovcreate(&iov_, mount_point.c_str(), buffer_size, 0, -1);
        if (ret != 0) {
            debug_log("Failed to create Iov: error %d", ret);
            throw std::runtime_error("Failed to create Iov: error " + std::to_string(ret));
        }
        debug_log("Iov created: base=%p, size=%lu", iov_.base, iov_.size);
        
        // Register Iov buffer as CUDA pinned memory for faster transfers
        cudaError_t cuda_err = cudaHostRegister(iov_.base, iov_.size, cudaHostRegisterDefault);
        if (cuda_err != cudaSuccess) {
            debug_log("WARNING: Failed to register Iov as pinned memory: %s", cudaGetErrorString(cuda_err));
            debug_log("Continuing without pinned memory optimization");
            // Don't throw - continue without pinned memory optimization
        } else {
            debug_log("Iov successfully registered as CUDA pinned memory");
        }
        
        // Create Ior (I/O ring)
        ret = hf3fs_iorcreate4(
            &ior_, 
            mount_point.c_str(), 
            entries, 
            true,  // for_read
            io_depth,
            0,     // timeout
            -1,    // numa
            0      // flags
        );
        if (ret != 0) {
            debug_log("Failed to create Ior: error %d", ret);
            hf3fs_iovdestroy(&iov_);
            throw std::runtime_error("Failed to create Ior: error " + std::to_string(ret));
        }
        debug_log("Ior created successfully");
        
        // Create CUDA stream for async memory operations
        cuda_err = cudaStreamCreate(&stream_);
        if (cuda_err != cudaSuccess) {
            debug_log("WARNING: Failed to create CUDA stream: %s", cudaGetErrorString(cuda_err));
            debug_log("Continuing without async stream - will use default stream");
            stream_ = nullptr;
        } else {
            debug_log("CUDA stream created successfully");
        }
    }
    
    ~ThreeFSReader() {
        debug_log("Destroying ThreeFSReader");
        // Cleanup in reverse order
        
        // Destroy CUDA stream
        if (stream_ != nullptr) {
            cudaError_t cuda_err = cudaStreamDestroy(stream_);
            if (cuda_err != cudaSuccess) {
                debug_log("WARNING: Failed to destroy CUDA stream: %s", cudaGetErrorString(cuda_err));
            } else {
                debug_log("CUDA stream destroyed successfully");
            }
        }
        
        if (ior_.iorh != nullptr) {
            debug_log("Destroying Ior");
            hf3fs_iordestroy(&ior_);
        }
        if (iov_.iovh != nullptr) {
            // Unregister CUDA pinned memory before destroying Iov
            cudaError_t cuda_err = cudaHostUnregister(iov_.base);
            if (cuda_err != cudaSuccess) {
                debug_log("WARNING: Failed to unregister pinned memory: %s", cudaGetErrorString(cuda_err));
                // Continue cleanup even if unregister fails
            } else {
                debug_log("Iov pinned memory unregistered successfully");
            }
            
            debug_log("Destroying Iov");
            hf3fs_iovdestroy(&iov_);
        }
        debug_log("ThreeFSReader destroyed");
    }
    
    // Prevent copying
    ThreeFSReader(const ThreeFSReader&) = delete;
    ThreeFSReader& operator=(const ThreeFSReader&) = delete;
    
    /**
     * Open a file and register it for USRBIO operations
     * 
     * @param path: File path
     * @param flags: OS open flags
     * @return: Registered file descriptor
     */
    int open(const std::string& path, int flags = O_RDONLY) {
        debug_log("Opening file: %s (flags=0x%x)", path.c_str(), flags);
        
        int fd = ::open(path.c_str(), flags, 0644);
        if (fd < 0) {
            debug_log("Failed to open file: %s (errno=%d)", strerror(errno), errno);
            throw std::runtime_error("Failed to open file: " + std::string(strerror(errno)));
        }
        debug_log("File opened: fd=%d", fd);
        
        // Note: hf3fs_reg_fd returns:
        //   - Negative value (< 0): Success, value is -dupfd
        //   - Positive value (> 0): Error, value is errno
        int ret = hf3fs_reg_fd(fd, 0);
        if (ret > 0) {  // Only positive values are errors
            debug_log("Failed to register fd %d: %s (ret=%d)", fd, strerror(ret), ret);
            ::close(fd);
            throw std::runtime_error("Failed to register fd: " + std::string(strerror(ret)));
        }
        debug_log("FD registered: fd=%d, dupfd=%d", fd, -ret);
        
        fd_map_[path] = fd;
        registered_fds_.push_back(fd);
        
        return fd;
    }
    
    /**
     * Close a registered file descriptor
     * 
     * @param fd: File descriptor to close
     */
    void close_fd(int fd) {
        debug_log("Closing fd: %d", fd);
        auto it = std::find(registered_fds_.begin(), registered_fds_.end(), fd);
        if (it != registered_fds_.end()) {
            hf3fs_dereg_fd(fd);
            registered_fds_.erase(it);
            ::close(fd);
            debug_log("FD closed: %d", fd);
        } else {
            debug_log("FD not found in registered list: %d", fd);
        }
    }
    
    /**
     * Close all registered file descriptors
     */
    void close_all() {
        debug_log("Closing all registered FDs (count=%zu)", registered_fds_.size());
        for (int fd : registered_fds_) {
            try {
                debug_log("Closing fd: %d", fd);
                hf3fs_dereg_fd(fd);
                ::close(fd);
            } catch (...) {
                debug_log("Error closing fd: %d", fd);
                // Ignore errors during cleanup
            }
        }
        registered_fds_.clear();
        fd_map_.clear();
        debug_log("All FDs closed");
    }
    
    /**
     * Get file descriptor for a path
     * 
     * @param path: File path
     * @return: File descriptor
     */
    int get_fd(const std::string& path) const {
        auto it = fd_map_.find(path);
        if (it == fd_map_.end()) {
            throw std::runtime_error("File not found: " + path);
        }
        return it->second;
    }
    
    /**
     * Submit an async read request
     * 
     * @param fd: File descriptor
     * @param dev_ptr: Device memory pointer (from gds_device_buffer.get_base_address())
     * @param offset: File offset
     * @param length: Read length
     * @param ptr_offset: Buffer offset
     * @return: Request ID for wait_read
     */
    int submit_read(int fd, uintptr_t dev_ptr, int64_t offset, int64_t length, int64_t ptr_offset) {
        auto t_start = now();
        
        // Allocate request ID
        int req_id = next_req_id_++;
        
        debug_log("submit_read: req_id=%d, fd=%d, offset=%ld, length=%ld, ptr_offset=%ld, dev_ptr=0x%lx",
                  req_id, fd, offset, length, ptr_offset, dev_ptr);
        
        // Prepare I/O request
        auto t_prep_start = now();
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            
            // Note: hf3fs_prep_io expects the absolute value of fd
            // because the fd registry uses abs(fd) as the key
            int abs_fd = (fd < 0) ? -fd : fd;
            
            uintptr_t dest_ptr = dev_ptr + ptr_offset;
            uintptr_t iov_base = reinterpret_cast<uintptr_t>(iov_.base);
            uintptr_t iov_end = iov_base + iov_.size;
            
            debug_log("Preparing I/O: abs_fd=%d, dest_ptr=0x%lx", abs_fd, dest_ptr);
            debug_log("  Iov range: base=0x%lx, end=0x%lx, size=%lu", iov_base, iov_end, iov_.size);
            debug_log("  Request: dest=0x%lx, length=%ld, dest_end=0x%lx", dest_ptr, length, dest_ptr + length);
            
            // Validate bounds before calling prep_io
            if (dest_ptr < iov_base) {
                debug_log("ERROR: dest_ptr (0x%lx) < iov_base (0x%lx)", dest_ptr, iov_base);
                throw std::runtime_error("dest_ptr below iov range");
            }
            if (dest_ptr + length > iov_end) {
                debug_log("ERROR: dest_ptr + length (0x%lx) > iov_end (0x%lx)", dest_ptr + length, iov_end);
                debug_log("  Overflow: need %lu bytes, but only %lu bytes available from dest_ptr to iov_end",
                         length, iov_end - dest_ptr);
                throw std::runtime_error("dest_ptr + length exceeds iov range: need " + 
                                       std::to_string(length) + " bytes, but only " + 
                                       std::to_string(iov_end - dest_ptr) + " bytes available. " +
                                       "Please split large reads into chunks <= " + 
                                       std::to_string(iov_.size) + " bytes (Iov size)");
            }
            
            debug_log("Bounds check passed: request fits within Iov");
            
            int ret = hf3fs_prep_io(
                &ior_,
                &iov_,
                true,   // read
                reinterpret_cast<void*>(dest_ptr),
                abs_fd,  // Use absolute value of fd
                static_cast<uint64_t>(offset),
                static_cast<uint64_t>(length),
                reinterpret_cast<const void*>(static_cast<uintptr_t>(req_id))
            );
            
            if (ret < 0) {
                debug_log("prep_io failed: ret=%d, fd=%d, abs_fd=%d, offset=%ld, length=%ld",
                         ret, fd, abs_fd, offset, length);
                throw std::runtime_error("prep_io failed: error " + std::to_string(ret) + 
                                       " (fd=" + std::to_string(fd) + 
                                       ", abs_fd=" + std::to_string(abs_fd) + 
                                       ", offset=" + std::to_string(offset) + 
                                       ", length=" + std::to_string(length) + ")");
            }
            
            debug_log("I/O prepared: io_idx=%d", ret);
            pending_[req_id] = std::make_unique<RequestContext>(offset, length, ptr_offset);
        }
        auto t_prep_end = now();
        
        // Submit the I/O request immediately
        // Note: hf3fs_submit_ios can be called multiple times to submit batched requests
        auto t_submit_start = now();
        debug_log("Calling hf3fs_submit_ios for req_id=%d", req_id);
        hf3fs_submit_ios(&ior_);
        auto t_submit_end = now();
        
        auto t_end = now();
        
        // Performance logging
        double prep_ms = elapsed_ms(t_prep_start, t_prep_end);
        double submit_ms = elapsed_ms(t_submit_start, t_submit_end);
        double total_ms = elapsed_ms(t_start, t_end);
        
        debug_log("submit_read timing: prep=%.3f ms, submit=%.3f ms, total=%.3f ms",
                 prep_ms, submit_ms, total_ms);
        
        debug_log("submit_read completed: req_id=%d", req_id);
        return req_id;
    }
    
    /**
     * Wait for a specific read request to complete
     * 
     * @param req_id: Request ID from submit_read
     * @return: Bytes read, or negative error code
     */
    int64_t wait_read(int req_id) {
        auto t_start = now();
        debug_log("wait_read: req_id=%d", req_id);
        
        // Wait for completion
        struct hf3fs_cqe cqe;
        int wait_iterations = 0;
        
        while (true) {
            {
                std::lock_guard<std::mutex> lock(pending_mutex_);
                auto it = pending_.find(req_id);
                if (it != pending_.end() && it->second->completed) {
                    int64_t result = it->second->result;
                    auto t_end = now();
                    double elapsed = elapsed_ms(t_start, t_end);
                    debug_log("wait_read: req_id=%d already completed, result=%ld, time=%.3f ms, iterations=%d",
                             req_id, result, elapsed, wait_iterations);
                    pending_.erase(it);
                    return result;
                }
            }
            
            // Wait for completions
            auto t_wait_start = now();
            debug_log("wait_read: calling hf3fs_wait_for_ios for req_id=%d (iteration %d)",
                     req_id, wait_iterations);
            int count = hf3fs_wait_for_ios(&ior_, &cqe, 1, 1, nullptr);
            auto t_wait_end = now();
            double wait_ms = elapsed_ms(t_wait_start, t_wait_end);
            
            if (count < 0) {
                debug_log("wait_for_ios failed: count=%d", count);
                throw std::runtime_error("wait_for_ios failed: error " + std::to_string(count));
            }
            
            debug_log("wait_for_ios returned: count=%d, time=%.3f ms", count, wait_ms);
            wait_iterations++;
            
            // Process completion
            {
                std::lock_guard<std::mutex> lock(pending_mutex_);
                int64_t completed_req_id = reinterpret_cast<int64_t>(cqe.userdata);
                auto it = pending_.find(completed_req_id);
                if (it != pending_.end()) {
                    it->second->completed = true;
                    it->second->result = cqe.result;
                    
                    debug_log("Completion received: req_id=%ld, result=%ld, index=%d",
                             completed_req_id, cqe.result, cqe.index);
                    
                    // If this is the requested one, return
                    if (completed_req_id == req_id) {
                        int64_t result = cqe.result;
                        auto t_end = now();
                        double total_ms = elapsed_ms(t_start, t_end);
                        debug_log("wait_read completed: req_id=%d, result=%ld, total_time=%.3f ms, iterations=%d",
                                 req_id, result, total_ms, wait_iterations);
                        pending_.erase(it);
                        return result;
                    } else {
                        debug_log("Received completion for different req_id: got=%ld, waiting_for=%d",
                                 completed_req_id, req_id);
                    }
                }
            }
        }
    }
    
    /**
     * Read file data in chunks to device memory
     * 
     * This is a high-level method that handles chunking automatically.
     * It reads data from file to Iov (shared memory), then copies to device memory.
     * 
     * @param fd: File descriptor
     * @param dev_ptr: Device memory pointer (destination)
     * @param file_offset: File offset to start reading
     * @param total_length: Total bytes to read
     * @param chunk_size: Size of each chunk (default: Iov size)
     * @param copy_func: Optional callback to copy from Iov to device memory
     *                   Signature: void(void* iov_ptr, void* dev_ptr, size_t size)
     * @return: Total bytes read
     */
    int64_t read_chunked(
        int fd,
        uintptr_t dev_ptr,
        int64_t file_offset,
        int64_t total_length,
        int64_t chunk_size = 0,
        std::function<void(void*, void*, size_t)> copy_func = nullptr
    ) {
        auto t_start = now();
        debug_log("read_chunked: fd=%d, dev_ptr=0x%lx, file_offset=%ld, total_length=%ld",
                  fd, dev_ptr, file_offset, total_length);
        
        // Use Iov size as default chunk size
        if (chunk_size <= 0) {
            chunk_size = iov_.size;
        }
        
        // Ensure chunk size doesn't exceed Iov size
        if (chunk_size > static_cast<int64_t>(iov_.size)) {
            debug_log("Adjusting chunk_size from %ld to %lu (Iov size)", chunk_size, iov_.size);
            chunk_size = iov_.size;
        }
        
        debug_log("Using chunk_size: %ld bytes", chunk_size);
        
        int64_t total_read = 0;
        int64_t remaining = total_length;
        int chunk_idx = 0;
        
        // Performance tracking
        double total_submit_ms = 0.0;
        double total_wait_ms = 0.0;
        double total_copy_ms = 0.0;
        
        while (remaining > 0) {
            int64_t current_chunk = std::min(chunk_size, remaining);
            int64_t current_file_offset = file_offset + total_read;
            
            debug_log("Chunk %d: reading %ld bytes from offset %ld",
                     chunk_idx, current_chunk, current_file_offset);
            
            // Step 1: Submit read request
            auto t_submit_start = now();
            int req_id = submit_read(
                fd,
                reinterpret_cast<uintptr_t>(iov_.base),  // Read to Iov
                current_file_offset,
                current_chunk,
                0  // Always use Iov base
            );
            auto t_submit_end = now();
            double submit_ms = elapsed_ms(t_submit_start, t_submit_end);
            total_submit_ms += submit_ms;
            perf_log(("Chunk " + std::to_string(chunk_idx) + " submit_read").c_str(), submit_ms);
            
            // Step 2: Wait for read to complete
            auto t_wait_start = now();
            int64_t bytes_read = wait_read(req_id);
            auto t_wait_end = now();
            double wait_ms = elapsed_ms(t_wait_start, t_wait_end);
            total_wait_ms += wait_ms;
            perf_log(("Chunk " + std::to_string(chunk_idx) + " wait_read").c_str(), wait_ms);
            
            if (bytes_read < 0) {
                debug_log("ERROR: Read failed with error %ld", bytes_read);
                throw std::runtime_error("read_chunked: read failed with error " + 
                                       std::to_string(bytes_read));
            }
            
            debug_log("Read completed: %ld bytes", bytes_read);
            
            // Step 3: Copy from Iov to device memory
            auto t_copy_start = now();
            if (copy_func) {
                debug_log("Calling custom copy function");
                copy_func(
                    iov_.base,
                    reinterpret_cast<void*>(dev_ptr + total_read),
                    bytes_read
                );
            } else {
                // Default: Use cudaMemcpy for GPU memory
                debug_log("Using cudaMemcpy: src=0x%lx, dst=0x%lx, size=%ld",
                         reinterpret_cast<uintptr_t>(iov_.base),
                         dev_ptr + total_read,
                         bytes_read);
                
                cudaError_t err = cudaMemcpy(
                    reinterpret_cast<void*>(dev_ptr + total_read),
                    iov_.base,
                    bytes_read,
                    cudaMemcpyHostToDevice
                );
                
                if (err != cudaSuccess) {
                    debug_log("ERROR: cudaMemcpy failed: %s", cudaGetErrorString(err));
                    throw std::runtime_error(std::string("cudaMemcpy failed: ") + 
                                           cudaGetErrorString(err));
                }
            }
            auto t_copy_end = now();
            double copy_ms = elapsed_ms(t_copy_start, t_copy_end);
            total_copy_ms += copy_ms;
            perf_log(("Chunk " + std::to_string(chunk_idx) + " cudaMemcpy").c_str(), copy_ms);
            
            total_read += bytes_read;
            remaining -= bytes_read;
            chunk_idx++;
            
            debug_log("Progress: %ld / %ld bytes (%.1f%%)",
                     total_read, total_length,
                     100.0 * total_read / total_length);
            
            // If we read less than requested, we've reached EOF
            if (bytes_read < current_chunk) {
                debug_log("Reached EOF: read %ld bytes, expected %ld bytes",
                         bytes_read, current_chunk);
                break;
            }
        }
        
        auto t_end = now();
        double total_ms = elapsed_ms(t_start, t_end);
        
        // Summary statistics
        if (is_debug_enabled()) {
            fprintf(stderr, "\n[ThreeFSReader PERF SUMMARY]\n");
            fprintf(stderr, "  Total chunks: %d\n", chunk_idx);
            fprintf(stderr, "  Total bytes: %ld (%.2f MB)\n", total_read, total_read / 1024.0 / 1024.0);
            fprintf(stderr, "  Total time: %.3f ms\n", total_ms);
            fprintf(stderr, "  Throughput: %.2f MB/s\n", (total_read / 1024.0 / 1024.0) / (total_ms / 1000.0));
            fprintf(stderr, "  Time breakdown:\n");
            fprintf(stderr, "    - submit_read: %.3f ms (%.1f%%)\n", total_submit_ms, 100.0 * total_submit_ms / total_ms);
            fprintf(stderr, "    - wait_read: %.3f ms (%.1f%%)\n", total_wait_ms, 100.0 * total_wait_ms / total_ms);
            fprintf(stderr, "    - cudaMemcpy: %.3f ms (%.1f%%)\n", total_copy_ms, 100.0 * total_copy_ms / total_ms);
            fprintf(stderr, "    - overhead: %.3f ms (%.1f%%)\n", 
                    total_ms - total_submit_ms - total_wait_ms - total_copy_ms,
                    100.0 * (total_ms - total_submit_ms - total_wait_ms - total_copy_ms) / total_ms);
            fprintf(stderr, "  Average per chunk:\n");
            fprintf(stderr, "    - submit_read: %.3f ms\n", total_submit_ms / chunk_idx);
            fprintf(stderr, "    - wait_read: %.3f ms\n", total_wait_ms / chunk_idx);
            fprintf(stderr, "    - cudaMemcpy: %.3f ms\n", total_copy_ms / chunk_idx);
            fflush(stderr);
        }
        
        debug_log("read_chunked completed: total %ld bytes read", total_read);
        return total_read;
    }
    
    /**
     * Read file data in chunks with pipelined double buffering
     * 
     * This method implements a pipelined approach where download and GPU copy
     * operations are overlapped using double buffering technique.
     * 
     * Pipeline workflow:
     * - Chunk 0: Download to Buffer A → Copy to GPU
     * - Chunk 1: Download to Buffer B (while copying chunk 0) → Copy to GPU
     * - Chunk 2: Download to Buffer A (while copying chunk 1) → Copy to GPU
     * - ...
     * 
     * @param fd: File descriptor
     * @param dev_ptr: Device memory pointer (destination)
     * @param file_offset: File offset to start reading
     * @param total_length: Total bytes to read
     * @param chunk_size: Size of each chunk (default: half of Iov size for double buffering)
     * @return: Total bytes read
     */
    int64_t read_chunked_pipelined(
        int fd,
        uintptr_t dev_ptr,
        int64_t file_offset,
        int64_t total_length,
        int64_t chunk_size = 0
    ) {
        auto t_start = now();
        debug_log("read_chunked_pipelined: fd=%d, dev_ptr=0x%lx, file_offset=%ld, total_length=%ld",
                  fd, dev_ptr, file_offset, total_length);
        
        // Calculate buffer size for double buffering (split Iov into 2 buffers)
        int64_t buffer_size = iov_.size / 2;
        
        // Use half of Iov size as default chunk size for double buffering
        if (chunk_size <= 0) {
            chunk_size = buffer_size;
        }
        
        // Ensure chunk size doesn't exceed buffer size
        if (chunk_size > buffer_size) {
            debug_log("Adjusting chunk_size from %ld to %ld (half of Iov size for double buffering)", 
                     chunk_size, buffer_size);
            chunk_size = buffer_size;
        }
        
        debug_log("Using chunk_size: %ld bytes, buffer_size: %ld bytes", chunk_size, buffer_size);
        
        // Buffer addresses
        void* buffer_a = iov_.base;
        void* buffer_b = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(iov_.base) + buffer_size);
        
        debug_log("Buffer A: %p, Buffer B: %p", buffer_a, buffer_b);
        
        int64_t total_read = 0;
        int64_t remaining = total_length;
        int chunk_idx = 0;
        
        // Performance tracking
        double total_submit_ms = 0.0;
        double total_wait_ms = 0.0;
        double total_copy_ms = 0.0;
        double total_sync_ms = 0.0;
        
        // Use default stream if custom stream creation failed
        cudaStream_t copy_stream = (stream_ != nullptr) ? stream_ : 0;
        
        // First chunk - warmup (no previous copy to wait for)
        if (remaining > 0) {
            int64_t current_chunk = std::min(chunk_size, remaining);
            int64_t current_file_offset = file_offset;
            
            debug_log("Chunk 0 (warmup): reading %ld bytes from offset %ld to Buffer A",
                     current_chunk, current_file_offset);
            
            // Submit read to Buffer A
            auto t_submit_start = now();
            int req_id = submit_read(
                fd,
                reinterpret_cast<uintptr_t>(buffer_a),
                current_file_offset,
                current_chunk,
                0
            );
            auto t_submit_end = now();
            double submit_ms = elapsed_ms(t_submit_start, t_submit_end);
            total_submit_ms += submit_ms;
            
            // Wait for download to complete
            auto t_wait_start = now();
            int64_t bytes_read = wait_read(req_id);
            auto t_wait_end = now();
            double wait_ms = elapsed_ms(t_wait_start, t_wait_end);
            total_wait_ms += wait_ms;
            
            if (bytes_read < 0) {
                debug_log("ERROR: Read failed with error %ld", bytes_read);
                throw std::runtime_error("read_chunked_pipelined: read failed with error " + 
                                       std::to_string(bytes_read));
            }
            
            // Start async copy to GPU
            auto t_copy_start = now();
            cudaError_t err = cudaMemcpyAsync(
                reinterpret_cast<void*>(dev_ptr),
                buffer_a,
                bytes_read,
                cudaMemcpyHostToDevice,
                copy_stream
            );
            auto t_copy_end = now();
            double copy_ms = elapsed_ms(t_copy_start, t_copy_end);
            total_copy_ms += copy_ms;
            
            if (err != cudaSuccess) {
                debug_log("ERROR: cudaMemcpyAsync failed: %s", cudaGetErrorString(err));
                throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") + 
                                       cudaGetErrorString(err));
            }
            
            total_read += bytes_read;
            remaining -= bytes_read;
            chunk_idx++;
            
            debug_log("Chunk 0 completed: %ld bytes, submit=%.3fms, wait=%.3fms, copy_start=%.3fms",
                     bytes_read, submit_ms, wait_ms, copy_ms);
            
            // If we read less than requested, we've reached EOF
            if (bytes_read < current_chunk) {
                debug_log("Reached EOF after first chunk");
                // Wait for the copy to complete before returning
                auto t_sync_start = now();
                cudaStreamSynchronize(copy_stream);
                auto t_sync_end = now();
                total_sync_ms += elapsed_ms(t_sync_start, t_sync_end);
                
                auto t_end = now();
                double total_ms = elapsed_ms(t_start, t_end);
                print_pipelined_summary(total_read, total_ms, chunk_idx, 
                                       total_submit_ms, total_wait_ms, total_copy_ms, total_sync_ms);
                return total_read;
            }
        }
        
        // Pipeline loop - process remaining chunks with double buffering
        while (remaining > 0) {
            int64_t current_chunk = std::min(chunk_size, remaining);
            int64_t current_file_offset = file_offset + total_read;
            
            // Determine which buffer to use (alternate between A and B)
            void* current_buffer = (chunk_idx % 2 == 0) ? buffer_a : buffer_b;
            const char* buffer_name = (chunk_idx % 2 == 0) ? "Buffer A" : "Buffer B";
            
            debug_log("Chunk %d: reading %ld bytes from offset %ld to %s",
                     chunk_idx, current_chunk, current_file_offset, buffer_name);
            
            // Step 1: Submit read for current chunk (to current buffer)
            auto t_submit_start = now();
            int req_id = submit_read(
                fd,
                reinterpret_cast<uintptr_t>(current_buffer),
                current_file_offset,
                current_chunk,
                0
            );
            auto t_submit_end = now();
            double submit_ms = elapsed_ms(t_submit_start, t_submit_end);
            total_submit_ms += submit_ms;
            
            // Step 2: Wait for previous chunk's GPU copy to complete
            // This ensures the previous buffer is free for reuse
            auto t_sync_start = now();
            cudaStreamSynchronize(copy_stream);
            auto t_sync_end = now();
            double sync_ms = elapsed_ms(t_sync_start, t_sync_end);
            total_sync_ms += sync_ms;
            
            debug_log("Previous copy synchronized: %.3fms", sync_ms);
            
            // Step 3: Wait for current chunk's download to complete
            auto t_wait_start = now();
            int64_t bytes_read = wait_read(req_id);
            auto t_wait_end = now();
            double wait_ms = elapsed_ms(t_wait_start, t_wait_end);
            total_wait_ms += wait_ms;
            
            if (bytes_read < 0) {
                debug_log("ERROR: Read failed with error %ld", bytes_read);
                throw std::runtime_error("read_chunked_pipelined: read failed with error " + 
                                       std::to_string(bytes_read));
            }
            
            debug_log("Download completed: %ld bytes", bytes_read);
            
            // Step 4: Start async copy for current chunk to GPU
            auto t_copy_start = now();
            cudaError_t err = cudaMemcpyAsync(
                reinterpret_cast<void*>(dev_ptr + total_read),
                current_buffer,
                bytes_read,
                cudaMemcpyHostToDevice,
                copy_stream
            );
            auto t_copy_end = now();
            double copy_ms = elapsed_ms(t_copy_start, t_copy_end);
            total_copy_ms += copy_ms;
            
            if (err != cudaSuccess) {
                debug_log("ERROR: cudaMemcpyAsync failed: %s", cudaGetErrorString(err));
                throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") + 
                                       cudaGetErrorString(err));
            }
            
            total_read += bytes_read;
            remaining -= bytes_read;
            chunk_idx++;
            
            debug_log("Chunk %d completed: submit=%.3fms, sync=%.3fms, wait=%.3fms, copy_start=%.3fms",
                     chunk_idx - 1, submit_ms, sync_ms, wait_ms, copy_ms);
            
            // If we read less than requested, we've reached EOF
            if (bytes_read < current_chunk) {
                debug_log("Reached EOF: read %ld bytes, expected %ld bytes",
                         bytes_read, current_chunk);
                break;
            }
        }
        
        // Wait for the last copy to complete
        debug_log("Waiting for final copy to complete");
        auto t_final_sync_start = now();
        cudaStreamSynchronize(copy_stream);
        auto t_final_sync_end = now();
        total_sync_ms += elapsed_ms(t_final_sync_start, t_final_sync_end);
        
        auto t_end = now();
        double total_ms = elapsed_ms(t_start, t_end);
        
        // Print performance summary
        print_pipelined_summary(total_read, total_ms, chunk_idx, 
                               total_submit_ms, total_wait_ms, total_copy_ms, total_sync_ms);
        
        debug_log("read_chunked_pipelined completed: total %ld bytes read", total_read);
        return total_read;
    }
    
    /**
     * Helper function to print pipelined performance summary
     */
    void print_pipelined_summary(
        int64_t total_read,
        double total_ms,
        int chunk_count,
        double total_submit_ms,
        double total_wait_ms,
        double total_copy_ms,
        double total_sync_ms
    ) {
        if (!is_debug_enabled()) return;
        fprintf(stderr, "\n[ThreeFSReader PERF SUMMARY - Pipelined Double Buffering]\n");
        fprintf(stderr, "  Total chunks: %d\n", chunk_count);
        fprintf(stderr, "  Total bytes: %ld (%.2f MB)\n", total_read, total_read / 1024.0 / 1024.0);
        fprintf(stderr, "  Total time: %.3f ms\n", total_ms);
        fprintf(stderr, "  Throughput: %.2f MB/s (%.2f GB/s)\n", 
                (total_read / 1024.0 / 1024.0) / (total_ms / 1000.0),
                (total_read / 1024.0 / 1024.0 / 1024.0) / (total_ms / 1000.0));
        fprintf(stderr, "  Time breakdown:\n");
        fprintf(stderr, "    - submit_read: %.3f ms (%.1f%%)\n", 
                total_submit_ms, 100.0 * total_submit_ms / total_ms);
        fprintf(stderr, "    - wait_read (download): %.3f ms (%.1f%%)\n", 
                total_wait_ms, 100.0 * total_wait_ms / total_ms);
        fprintf(stderr, "    - cudaMemcpyAsync (start): %.3f ms (%.1f%%)\n", 
                total_copy_ms, 100.0 * total_copy_ms / total_ms);
        fprintf(stderr, "    - cudaStreamSynchronize: %.3f ms (%.1f%%)\n", 
                total_sync_ms, 100.0 * total_sync_ms / total_ms);
        fprintf(stderr, "    - overhead: %.3f ms (%.1f%%)\n", 
                total_ms - total_submit_ms - total_wait_ms - total_copy_ms - total_sync_ms,
                100.0 * (total_ms - total_submit_ms - total_wait_ms - total_copy_ms - total_sync_ms) / total_ms);
        fprintf(stderr, "  Average per chunk:\n");
        fprintf(stderr, "    - submit_read: %.3f ms\n", total_submit_ms / chunk_count);
        fprintf(stderr, "    - wait_read (download): %.3f ms\n", total_wait_ms / chunk_count);
        fprintf(stderr, "    - cudaMemcpyAsync (start): %.3f ms\n", total_copy_ms / chunk_count);
        fprintf(stderr, "    - cudaStreamSynchronize: %.3f ms\n", total_sync_ms / chunk_count);
        fprintf(stderr, "  Pipeline efficiency:\n");
        fprintf(stderr, "    - Download time: %.3f ms\n", total_wait_ms);
        fprintf(stderr, "    - Copy sync time: %.3f ms\n", total_sync_ms);
        fprintf(stderr, "    - Overlap benefit: %.3f ms (%.1f%% of download time hidden)\n",
                total_wait_ms - total_sync_ms,
                100.0 * (total_wait_ms - total_sync_ms) / total_wait_ms);
        fflush(stderr);
    }
    
    /**
     * Read file data to Iov only (no GPU copy) - for testing pure download performance
     * 
     * This method only downloads data from 3FS to Iov (shared memory), without copying to GPU.
     * Use this to test the pure network download bandwidth limit.
     * 
     * @param fd: File descriptor
     * @param file_offset: File offset to start reading
     * @param total_length: Total bytes to read
     * @param chunk_size: Size of each chunk (default: Iov size)
     * @return: Total bytes read
     */
    int64_t read_to_iov_only(
        int fd,
        int64_t file_offset,
        int64_t total_length,
        int64_t chunk_size = 0
    ) {
        auto t_start = now();
        debug_log("read_to_iov_only: fd=%d, file_offset=%ld, total_length=%ld",
                  fd, file_offset, total_length);
        
        // Use Iov size as default chunk size
        if (chunk_size <= 0) {
            chunk_size = iov_.size;
        }
        
        // Ensure chunk size doesn't exceed Iov size
        if (chunk_size > static_cast<int64_t>(iov_.size)) {
            debug_log("Adjusting chunk_size from %ld to %lu (Iov size)", chunk_size, iov_.size);
            chunk_size = iov_.size;
        }
        
        debug_log("Using chunk_size: %ld bytes", chunk_size);
        
        int64_t total_read = 0;
        int64_t remaining = total_length;
        int chunk_idx = 0;
        
        // Performance tracking
        double total_submit_ms = 0.0;
        double total_wait_ms = 0.0;
        
        while (remaining > 0) {
            int64_t current_chunk = std::min(chunk_size, remaining);
            int64_t current_file_offset = file_offset + total_read;
            
            debug_log("Chunk %d: reading %ld bytes from offset %ld",
                     chunk_idx, current_chunk, current_file_offset);
            
            // Step 1: Submit read request
            auto t_submit_start = now();
            int req_id = submit_read(
                fd,
                reinterpret_cast<uintptr_t>(iov_.base),  // Read to Iov
                current_file_offset,
                current_chunk,
                0  // Always use Iov base
            );
            auto t_submit_end = now();
            double submit_ms = elapsed_ms(t_submit_start, t_submit_end);
            total_submit_ms += submit_ms;
            
            // Step 2: Wait for read to complete
            auto t_wait_start = now();
            int64_t bytes_read = wait_read(req_id);
            auto t_wait_end = now();
            double wait_ms = elapsed_ms(t_wait_start, t_wait_end);
            total_wait_ms += wait_ms;
            
            if (bytes_read < 0) {
                debug_log("ERROR: Read failed with error %ld", bytes_read);
                throw std::runtime_error("read_to_iov_only: read failed with error " + 
                                       std::to_string(bytes_read));
            }
            
            debug_log("Read completed: %ld bytes", bytes_read);
            
            // No GPU copy - data stays in Iov
            
            total_read += bytes_read;
            remaining -= bytes_read;
            chunk_idx++;
            
            debug_log("Progress: %ld / %ld bytes (%.1f%%)",
                     total_read, total_length,
                     100.0 * total_read / total_length);
            
            // If we read less than requested, we've reached EOF
            if (bytes_read < current_chunk) {
                debug_log("Reached EOF: read %ld bytes, expected %ld bytes",
                         bytes_read, current_chunk);
                break;
            }
        }
        
        auto t_end = now();
        double total_ms = elapsed_ms(t_start, t_end);
        
        // Summary statistics (pure download performance)
        if (is_debug_enabled()) {
            fprintf(stderr, "\n[ThreeFSReader PERF SUMMARY - Pure Download]\n");
            fprintf(stderr, "  Total chunks: %d\n", chunk_idx);
            fprintf(stderr, "  Total bytes: %ld (%.2f MB)\n", total_read, total_read / 1024.0 / 1024.0);
            fprintf(stderr, "  Total time: %.3f ms\n", total_ms);
            fprintf(stderr, "  Throughput: %.2f MB/s (%.2f GB/s)\n", 
                    (total_read / 1024.0 / 1024.0) / (total_ms / 1000.0),
                    (total_read / 1024.0 / 1024.0 / 1024.0) / (total_ms / 1000.0));
            fprintf(stderr, "  Time breakdown:\n");
            fprintf(stderr, "    - submit_read: %.3f ms (%.1f%%)\n", total_submit_ms, 100.0 * total_submit_ms / total_ms);
            fprintf(stderr, "    - wait_read (download): %.3f ms (%.1f%%)\n", total_wait_ms, 100.0 * total_wait_ms / total_ms);
            fprintf(stderr, "    - overhead: %.3f ms (%.1f%%)\n", 
                    total_ms - total_submit_ms - total_wait_ms,
                    100.0 * (total_ms - total_submit_ms - total_wait_ms) / total_ms);
            fprintf(stderr, "  Average per chunk:\n");
            fprintf(stderr, "    - submit_read: %.3f ms\n", total_submit_ms / chunk_idx);
            fprintf(stderr, "    - wait_read (download): %.3f ms\n", total_wait_ms / chunk_idx);
            fprintf(stderr, "  Note: This is PURE DOWNLOAD performance (no GPU copy)\n");
            fflush(stderr);
        }
        
        debug_log("read_to_iov_only completed: total %ld bytes read", total_read);
        return total_read;
    }
    
    /**
     * Wait for all pending requests to complete
     * 
     * @return: Vector of (req_id, byte_count) pairs
     */
    std::vector<std::pair<int, int64_t>> wait_all() {
        debug_log("wait_all: starting");
        
        std::vector<std::pair<int, int64_t>> results;
        std::vector<struct hf3fs_cqe> cqes(entries_);
        
        while (true) {
            int count;
            {
                std::lock_guard<std::mutex> lock(pending_mutex_);
                if (pending_.empty()) {
                    debug_log("wait_all: no pending requests, done");
                    break;
                }
                debug_log("wait_all: pending requests count=%zu", pending_.size());
            }
            
            count = hf3fs_wait_for_ios(&ior_, cqes.data(), entries_, 1, nullptr);
            if (count < 0) {
                debug_log("wait_for_ios failed: count=%d", count);
                throw std::runtime_error("wait_for_ios failed: error " + std::to_string(count));
            }
            
            debug_log("wait_all: received %d completions", count);
            
            // Process completions
            std::lock_guard<std::mutex> lock(pending_mutex_);
            for (int i = 0; i < count; ++i) {
                int64_t req_id = reinterpret_cast<int64_t>(cqes[i].userdata);
                auto it = pending_.find(req_id);
                if (it != pending_.end()) {
                    debug_log("wait_all: completion for req_id=%ld, result=%ld", req_id, cqes[i].result);
                    results.emplace_back(static_cast<int>(req_id), cqes[i].result);
                    pending_.erase(it);
                }
            }
        }
        
        debug_log("wait_all: completed, total results=%zu", results.size());
        return results;
    }
    
    /**
     * Get Iov base address for direct buffer access
     */
    uintptr_t get_iov_base() const {
        return reinterpret_cast<uintptr_t>(iov_.base);
    }
    
    /**
     * Get Iov length
     */
    uint64_t get_iov_length() const {
        return iov_.size;
    }
    
    /**
     * Get mount point
     */
    const std::string& get_mount_point() const {
        return mount_point_;
    }
    
    /**
     * Batch open files and read their SafeTensors headers using a thread pool.
     *
     * Each thread independently performs: open → fstat → hf3fs_reg_fd → read(8 bytes)
     * → parse header length → read(header JSON) → record fd.
     * This avoids USRBIO Ior queue limitations and leverages OS-level I/O parallelism.
     *
     * @param paths: List of file paths to open and read headers from
     * @param num_threads: Number of threads in the pool (default: 8)
     * @return: Vector of (path, header_json_string, header_length, file_size) tuples.
     *          header_length = n + 8 (8-byte length prefix + n-byte JSON).
     */
    std::vector<std::tuple<std::string, std::string, int64_t, int64_t>>
    open_and_read_headers(const std::vector<std::string>& paths, int num_threads = 8) {
        auto t_start = now();
        const size_t total_files = paths.size();
        debug_log("open_and_read_headers: %zu files, %d threads", total_files, num_threads);

        if (total_files == 0) {
            return {};
        }

        // Per-file result structure
        struct FileResult {
            std::string path;
            std::string header_json;
            int64_t header_length = 0;  // n + 8
            int64_t file_size = 0;
            int fd = -1;
            std::string error;
        };

        std::vector<FileResult> results(total_files);

        // Use thread pool for parallel open + read
        {
            BS::thread_pool pool(static_cast<unsigned int>(
                std::min(static_cast<int>(total_files), num_threads)));

            auto futures = pool.submit_sequence(
                static_cast<size_t>(0), total_files,
                [&](size_t idx) {
                    FileResult& result = results[idx];
                    result.path = paths[idx];

                    // Step 1: open file
                    int fd = ::open(paths[idx].c_str(), O_RDONLY, 0644);
                    if (fd < 0) {
                        result.error = "open failed: " + std::string(strerror(errno));
                        return;
                    }

                    // Step 2: fstat to get file size
                    struct stat st;
                    if (::fstat(fd, &st) < 0) {
                        result.error = "fstat failed: " + std::string(strerror(errno));
                        ::close(fd);
                        return;
                    }
                    result.file_size = st.st_size;

                    if (result.file_size < 8) {
                        result.error = "file too small: " + std::to_string(result.file_size);
                        ::close(fd);
                        return;
                    }

                    // Step 3: register fd with 3FS
                    int reg_ret = hf3fs_reg_fd(fd, 0);
                    if (reg_ret > 0) {
                        result.error = "hf3fs_reg_fd failed: " + std::string(strerror(reg_ret));
                        ::close(fd);
                        return;
                    }

                    // Step 4: read first 8 bytes (header length)
                    uint8_t len_buf[8];
                    ssize_t bytes_read = ::read(fd, len_buf, 8);
                    if (bytes_read != 8) {
                        result.error = "read header length failed: got " +
                                       std::to_string(bytes_read) + " bytes";
                        hf3fs_dereg_fd(fd);
                        ::close(fd);
                        return;
                    }

                    // Parse little-endian uint64
                    uint64_t header_size = 0;
                    for (int i = 7; i >= 0; --i) {
                        header_size = (header_size << 8) | len_buf[i];
                    }

                    if (header_size > 100000000ULL) {
                        result.error = "header too large: " + std::to_string(header_size);
                        hf3fs_dereg_fd(fd);
                        ::close(fd);
                        return;
                    }
                    if (static_cast<int64_t>(header_size) > result.file_size - 8) {
                        result.error = "invalid header length: n=" +
                                       std::to_string(header_size) +
                                       ", file_size=" + std::to_string(result.file_size);
                        hf3fs_dereg_fd(fd);
                        ::close(fd);
                        return;
                    }

                    // Step 5: read header JSON
                    std::string header_json(header_size, '\0');
                    size_t total_header_read = 0;
                    while (total_header_read < header_size) {
                        ssize_t n = ::read(fd, &header_json[total_header_read],
                                           header_size - total_header_read);
                        if (n <= 0) {
                            result.error = "read header JSON failed at offset " +
                                           std::to_string(total_header_read);
                            hf3fs_dereg_fd(fd);
                            ::close(fd);
                            return;
                        }
                        total_header_read += static_cast<size_t>(n);
                    }

                    result.header_json = std::move(header_json);
                    result.header_length = static_cast<int64_t>(header_size) + 8;
                    result.fd = fd;
                });

            futures.wait();
        }  // thread pool destroyed here, all threads joined

        // Collect results and register fds (single-threaded, under lock)
        std::vector<std::tuple<std::string, std::string, int64_t, int64_t>> output;
        output.reserve(total_files);
        int success_count = 0;
        int error_count = 0;

        for (auto& result : results) {
            if (!result.error.empty()) {
                debug_log("open_and_read_headers: FAILED %s: %s",
                         result.path.c_str(), result.error.c_str());
                error_count++;
                continue;
            }

            // Register fd in our tracking maps
            {
                std::lock_guard<std::mutex> lock(fd_mutex_);
                fd_map_[result.path] = result.fd;
                registered_fds_.push_back(result.fd);
            }

            output.emplace_back(
                result.path,
                std::move(result.header_json),
                result.header_length,
                result.file_size
            );
            success_count++;
        }

        auto t_end = now();
        double total_ms = elapsed_ms(t_start, t_end);

        if (is_debug_enabled()) {
            fprintf(stderr,
                    "[ThreeFSReader] open_and_read_headers: %d/%zu files succeeded, "
                    "%d failed, %.3f ms (%.3f ms/file)\n",
                    success_count, total_files, error_count,
                    total_ms, total_ms / total_files);
            fflush(stderr);
        }

        if (success_count == 0 && total_files > 0) {
            throw std::runtime_error(
                "open_and_read_headers: all " + std::to_string(total_files) +
                " files failed. First error: " + results[0].error);
        }

        return output;
    }

    /**
     * Check if library is available
     */
    static bool check_library() {
        // Try to create a minimal reader to check library availability
        // This is a simple check - in production, might want more robust detection
        return true;  // Library loading will fail at construction if not available
    }

private:
    std::string mount_point_;
    int entries_;
    int io_depth_;
    uint64_t buffer_size_;
    
    hf3fs_iov iov_;
    hf3fs_ior ior_;
    
    std::map<std::string, int> fd_map_;
    std::vector<int> registered_fds_;
    
    std::atomic<int> next_req_id_;
    
    std::mutex pending_mutex_;
    std::map<int, std::unique_ptr<RequestContext>> pending_;
    
    // Mutex for fd_map_ and registered_fds_ (used by open_and_read_headers)
    std::mutex fd_mutex_;
    
    // CUDA Stream for async memory copy
    cudaStream_t stream_;
};

// =============================================================================
// Python Module Definition
// =============================================================================

PYBIND11_MODULE(threefs_reader, m) {
    m.doc() = "3FS USRBIO High-Performance File Reader";
    
    // ThreeFSReader class
    py::class_<ThreeFSReader>(m, "ThreeFSReader")
        .def(py::init<const std::string&, int, int, uint64_t>(),
             py::arg("mount_point"),
             py::arg("entries") = 64,
             py::arg("io_depth") = 0,
             py::arg("buffer_size") = 1073741824ULL)  // 1GB default
        .def("open", &ThreeFSReader::open,
             py::arg("path"),
             py::arg("flags") = O_RDONLY)
        .def("close_fd", &ThreeFSReader::close_fd,
             py::arg("fd"))
        .def("close_all", &ThreeFSReader::close_all)
        .def("get_fd", &ThreeFSReader::get_fd,
             py::arg("path"))
        .def("submit_read", &ThreeFSReader::submit_read,
             py::arg("fd"),
             py::arg("dev_ptr"),
             py::arg("offset"),
             py::arg("length"),
             py::arg("ptr_offset"),
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("wait_read", &ThreeFSReader::wait_read,
             py::arg("req_id"),
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("wait_all", &ThreeFSReader::wait_all,
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("read_chunked", 
             [](ThreeFSReader& self, int fd, uintptr_t dev_ptr, int64_t file_offset, 
                int64_t total_length, int64_t chunk_size) {
                 return self.read_chunked(fd, dev_ptr, file_offset, total_length, chunk_size, nullptr);
             },
             py::arg("fd"),
             py::arg("dev_ptr"),
             py::arg("file_offset"),
             py::arg("total_length"),
             py::arg("chunk_size") = 0,
             pybind11::call_guard<pybind11::gil_scoped_release>(),
             "Read file data in chunks, automatically handling Iov buffer reuse")
        .def("read_to_iov_only", &ThreeFSReader::read_to_iov_only,
             py::arg("fd"),
             py::arg("file_offset"),
             py::arg("total_length"),
             py::arg("chunk_size") = 0,
             pybind11::call_guard<pybind11::gil_scoped_release>(),
             "Read file data to Iov only (no GPU copy) - for testing pure download performance")
        .def("read_chunked_pipelined",
             [](ThreeFSReader& self, int fd, uintptr_t dev_ptr, int64_t file_offset,
                int64_t total_length, int64_t chunk_size) {
                 return self.read_chunked_pipelined(fd, dev_ptr, file_offset, total_length, chunk_size);
             },
             py::arg("fd"),
             py::arg("dev_ptr"),
             py::arg("file_offset"),
             py::arg("total_length"),
             py::arg("chunk_size") = 0,
             pybind11::call_guard<pybind11::gil_scoped_release>(),
             "Read file data in chunks with pipelined double buffering for better performance")
        .def("open_and_read_headers", &ThreeFSReader::open_and_read_headers,
             py::arg("paths"),
             py::arg("num_threads") = 8,
             pybind11::call_guard<pybind11::gil_scoped_release>(),
             "Batch open files and read SafeTensors headers using thread pool")
        .def_property_readonly("iov_base", &ThreeFSReader::get_iov_base)
        .def_property_readonly("iov_length", &ThreeFSReader::get_iov_length)
        .def_property_readonly("mount_point", &ThreeFSReader::get_mount_point)
        .def_static("check_library", &ThreeFSReader::check_library);
    
    // Module-level functions
    m.def("check_library", &ThreeFSReader::check_library,
          "Check if 3FS USRBIO library is available");
}
