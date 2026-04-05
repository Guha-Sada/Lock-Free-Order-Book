#pragma once

#include <atomic>
#include <array>
#include <cstddef>
#include <type_traits>

// ---------------------------------------------------------------------------
// SPSCQueue<T, N>  —  Single-Producer Single-Consumer lock-free ring buffer
//
// Transfers objects of type T from exactly ONE producer thread to exactly ONE
// consumer thread without any mutex or kernel involvement.
//
// Template parameters:
//   T  — element type; must be trivially copyable (no heap allocation on copy)
//   N  — ring buffer capacity; MUST be a power of two
//        Usable capacity is N-1 (one slot kept empty to distinguish
//        "full" from "empty" without a separate count variable).
//
// Memory ordering rationale (this is a common interview topic):
//
//   write_pos_ is written only by the producer and read by both sides.
//   read_pos_  is written only by the consumer and read by both sides.
//
//   push():
//     - load(write_pos_) relaxed  — only we write it; we don't need to
//                                   synchronise with ourselves.
//     - load(read_pos_)  acquire  — we must *see* the consumer's latest
//                                   read_pos_ store so we don't overwrite
//                                   a slot the consumer hasn't finished with.
//     - store(write_pos_) release — makes the written item visible to the
//                                   consumer before it sees the updated index.
//
//   pop() is the mirror image for the consumer.
//
//   seq_cst is intentionally avoided: it inserts a full memory fence (MFENCE
//   on x86) on every operation, adding ~10–20 ns per call at high frequency.
//
// Cache-line separation:
//   write_pos_ and read_pos_ are placed on SEPARATE 64-byte cache lines via
//   alignas(64).  Without this, both atomics sit in the same cache line,
//   which bounces between producer and consumer cores on every access —
//   "false sharing".  The benchmark in bench_main.cpp demonstrates the
//   ~4x throughput difference this padding produces.
//
// Usage contract:
//   push() may only be called from one thread at a time (the producer).
//   pop()  may only be called from one thread at a time (the consumer).
//   Violating this is undefined behaviour — use SPMCQueue for fan-out.
// ---------------------------------------------------------------------------

template<typename T, size_t N>
class SPSCQueue {
    static_assert((N & (N - 1)) == 0,
                  "SPSCQueue: N must be a power of two (e.g. 1024, 4096)");
    static_assert(std::is_trivially_copyable_v<T>,
                  "SPSCQueue: T must be trivially copyable to avoid allocation in push/pop");

    static constexpr size_t MASK = N - 1;   // bitmask for fast modulo

public:
    SPSCQueue() = default;

    // Disable copy and move — the atomic members and fixed buffer make
    // these semantically wrong and physically dangerous.
    SPSCQueue(const SPSCQueue&)            = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;
    SPSCQueue(SPSCQueue&&)                 = delete;
    SPSCQueue& operator=(SPSCQueue&&)      = delete;

    // -----------------------------------------------------------------------
    // Producer interface (call only from the producer thread)
    // -----------------------------------------------------------------------

    // Attempts to enqueue item.  Returns true on success, false if full.
    // Never blocks; the caller must handle backpressure (spin, drop, etc.).
    bool push(const T& item) noexcept {
        const size_t write = write_pos_.load(std::memory_order_relaxed);
        const size_t next  = (write + 1) & MASK;

        // Full: the next write slot is the one the consumer is about to read.
        if (next == read_pos_.load(std::memory_order_acquire))
            return false;

        buffer_[write] = item;

        // Release: ensures the buffer write above is visible to the consumer
        // before it observes the updated write_pos_.
        write_pos_.store(next, std::memory_order_release);
        return true;
    }

    // -----------------------------------------------------------------------
    // Consumer interface (call only from the consumer thread)
    // -----------------------------------------------------------------------

    // Attempts to dequeue into item.  Returns true on success, false if empty.
    bool pop(T& item) noexcept {
        const size_t read = read_pos_.load(std::memory_order_relaxed);

        // Empty: consumer has caught up with producer.
        if (read == write_pos_.load(std::memory_order_acquire))
            return false;

        item = buffer_[read];

        // Release: allows the producer to reuse this slot after the store.
        read_pos_.store((read + 1) & MASK, std::memory_order_release);
        return true;
    }

    // -----------------------------------------------------------------------
    // Observers (approximate — values may be stale by the time you read them)
    // -----------------------------------------------------------------------

    bool empty() const noexcept {
        return read_pos_.load(std::memory_order_acquire)
            == write_pos_.load(std::memory_order_acquire);
    }

    // Returns the number of items currently in the queue.
    // Result is approximate in a live concurrent scenario.
    size_t size_approx() const noexcept {
        const size_t w = write_pos_.load(std::memory_order_acquire);
        const size_t r = read_pos_.load(std::memory_order_acquire);
        return (w - r) & MASK;
    }

    static constexpr size_t capacity() noexcept { return N - 1; }

private:
    // Producer-owned index — sits on its own cache line to prevent
    // false-sharing with read_pos_ (which the consumer owns).
    alignas(64) std::atomic<size_t> write_pos_{0};

    // Consumer-owned index — separate cache line.
    alignas(64) std::atomic<size_t> read_pos_{0};

    // The ring buffer itself.  Placed after the atomics so the atomics
    // are guaranteed to be on different cache lines from each other.
    std::array<T, N> buffer_{};
};
