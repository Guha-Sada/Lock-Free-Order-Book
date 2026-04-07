#pragma once

#include <atomic>
#include <array>
#include <cstddef>
#include <type_traits>

// ---------------------------------------------------------------------------
// NaiveSPSCQueue<T, N>  —  intentionally broken SPSC queue for benchmarking
//
// This is a byte-for-byte copy of SPSCQueue with ONE difference:
//   write_pos_ and read_pos_ are NOT separated onto different cache lines.
//
// They are declared as plain members with no alignas(64), so the compiler
// places them consecutively in memory.  On a 64-byte cache line, both fit
// comfortably in the same line (2 × 8 bytes = 16 bytes).
//
// WHY THIS IS A PROBLEM — false sharing:
//
//   The producer owns write_pos_ (only it writes to it).
//   The consumer owns read_pos_  (only it writes to it).
//
//   When the producer writes to write_pos_, the CPU must acquire exclusive
//   ownership of the cache line containing it.  This invalidates the
//   consumer's cached copy of that entire cache line — including read_pos_,
//   which the producer never touched.
//
//   The next time the consumer reads or writes read_pos_, it must fetch the
//   cache line from the producer's core.  The producer then has to do the
//   same in reverse.  The cache line bounces between cores on every single
//   push/pop pair, even though the two threads are accessing different bytes.
//
//   This is called false sharing: the sharing of a cache line between two
//   threads that logically have nothing to do with each other.
//
// EXPECTED BENCHMARK RESULT:
//
//   BM_SPSCQueue_Throughput      (padded)  ~3–5x higher throughput
//   BM_NaiveSPSCQueue_Throughput (naive)   much lower — cache line bounces
//
//   The exact ratio depends on your CPU's inter-core latency (NUMA distance,
//   ring bus vs mesh topology, etc.).  On a typical modern desktop CPU the
//   false-sharing version is 3–8x slower for the two-thread throughput test.
//   The single-threaded roundtrip benchmark shows almost no difference
//   because there is no second core to cause the bounce — which is why
//   the two-thread test is necessary to demonstrate the problem.
//
// DO NOT use this class in production.  It exists solely to make the
// false-sharing penalty measurable and visible.
// ---------------------------------------------------------------------------

template<typename T, size_t N>
class NaiveSPSCQueue {
    static_assert((N & (N - 1)) == 0,
                  "NaiveSPSCQueue: N must be a power of two");
    static_assert(std::is_trivially_copyable_v<T>,
                  "NaiveSPSCQueue: T must be trivially copyable");

    static constexpr size_t MASK = N - 1;

public:
    NaiveSPSCQueue() = default;

    NaiveSPSCQueue(const NaiveSPSCQueue&)            = delete;
    NaiveSPSCQueue& operator=(const NaiveSPSCQueue&) = delete;
    NaiveSPSCQueue(NaiveSPSCQueue&&)                 = delete;
    NaiveSPSCQueue& operator=(NaiveSPSCQueue&&)      = delete;

    bool push(const T& item) noexcept {
        const size_t write = write_pos_.load(std::memory_order_relaxed);
        const size_t next  = (write + 1) & MASK;
        if (next == read_pos_.load(std::memory_order_acquire))
            return false;
        buffer_[write] = item;
        write_pos_.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T& item) noexcept {
        const size_t read = read_pos_.load(std::memory_order_relaxed);
        if (read == write_pos_.load(std::memory_order_acquire))
            return false;
        item = buffer_[read];
        read_pos_.store((read + 1) & MASK, std::memory_order_release);
        return true;
    }

    bool empty() const noexcept {
        return read_pos_.load(std::memory_order_acquire)
            == write_pos_.load(std::memory_order_acquire);
    }

    static constexpr size_t capacity() noexcept { return N - 1; }

private:
    // -----------------------------------------------------------------------
    // THE ONLY DIFFERENCE FROM SPSCQueue:
    //
    // No alignas(64) here.  Both atomics land in the same cache line.
    // On a typical x86-64 system sizeof(atomic<size_t>) == 8, so these
    // two members occupy bytes 0–15 of the object — well within one 64-byte
    // cache line.
    //
    // Compare to SPSCQueue where:
    //   alignas(64) write_pos_  — occupies bytes 0–63   (its own cache line)
    //   alignas(64) read_pos_   — occupies bytes 64–127 (its own cache line)
    // -----------------------------------------------------------------------
    std::atomic<size_t> write_pos_{0};   // no alignas — shares cache line
    std::atomic<size_t> read_pos_{0};    // no alignas — same cache line as above

    std::array<T, N> buffer_{};
};
