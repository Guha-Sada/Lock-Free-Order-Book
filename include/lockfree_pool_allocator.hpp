#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <stdexcept>

#include "order.hpp"

// ---------------------------------------------------------------------------
// LockFreePoolAllocator  (Version B — ABA-safe lock-free free list)
//
// Extends the Version A design (see pool_allocator.hpp) to support concurrent
// allocate() and deallocate() calls from multiple threads.  The single-
// threaded free_head_ pointer is replaced with a 128-bit atomic TaggedPtr
// that pairs a pointer with a monotonically increasing version counter.
//
// WHY 128 BITS?
//   A naive lock-free free list uses a single atomic pointer and a CAS loop:
//
//     pop:  read head → set new_head = head->next → CAS(head, new_head)
//     push: read head → set node->next = head    → CAS(head, node)
//
//   This works correctly in the absence of one specific bug: the ABA problem.
//
// THE ABA PROBLEM (read this carefully — it comes up in interviews):
//
//   Suppose the free list is:  head → X → Y → Z
//
//   Thread A starts a pop:
//     1. Reads head = X.  Plans to CAS head from X to X->next (Y).
//     2. Gets preempted before the CAS.
//
//   Thread B runs while A is suspended:
//     3. Pops X.  head = Y → Z.
//     4. Pops Y.  head = Z.
//     5. Pushes X back.  head = X → Z.  (X->next is now Z, not Y)
//
//   Thread A resumes:
//     6. CAS(head, X, Y) — the address of head is still X, so the CAS
//        *succeeds*.  But X->next is now Z, not Y.  Thread A sets
//        head = Y (a freed node).  The free list is now corrupted.
//
//   The address went A → B → A (ABA), and the CAS couldn't tell.
//
// THE TAGGED POINTER SOLUTION:
//
//   Store head as a (pointer, version_counter) pair.  Every push increments
//   the version counter.  In step 5 above, Thread B pushes X back with
//   version=1, not version=0.  Thread A's CAS expects (X, version=0) but
//   finds (X, version=1), so the CAS fails and Thread A retries correctly.
//
// HARDWARE SUPPORT — platform-dependent:
//
//   x86-64: maps to CMPXCHG16B — a single instruction that atomically
//     compares and swaps a 16-byte memory location.  Requires the address
//     to be 16-byte aligned and the compiler flag -mcx16 (GCC) or nothing
//     extra on Clang.  is_lock_free() returns true.
//
//   ARM64 (Apple Silicon, AWS Graviton): uses LDXP/STXP (load/store
//     exclusive pair) in a LL/SC retry loop.  GCC requires linking with
//     -latomic; Clang handles it natively.  is_lock_free() may return false
//     under GCC but the code is still correct and contention-safe.
//
//   The constructor checks is_lock_free() at runtime and prints a diagnostic.
//   If it returns false, the atomic falls back to a library-level lock —
//   functionally correct but not truly lock-free.  This matters for
//   performance analysis but not correctness.
//
// MEMORY ORDERING:
//
//   push (deallocate) uses memory_order_release on CAS success.
//     — The write to node->next happens before the CAS.  The release
//       ordering ensures that any thread which then reads this node via
//       a successful pop will see the correct next pointer.
//
//   pop (allocate) uses memory_order_acquire on CAS success.
//     — Pairs with the release in push.  Guarantees we see all writes
//       that happened before the node was pushed (including node->next).
//
//   memory_order_relaxed on failure — we just retry; no sync needed.
//
// THREAD SAFETY:
//   All public methods are safe to call concurrently from any number of
//   threads.  Unlike Version A, there is no single "owning" thread.
//
// PERFORMANCE VS VERSION A:
//   Version B is intentionally slower than Version A for single-threaded
//   use: CMPXCHG16B costs ~8–15 ns vs ~3 ns for a plain pointer swap.
//   Under contention (multiple threads competing), failed CAS attempts
//   add retries.  Choose Version A when the allocator is owned by one
//   thread.  Choose Version B when true concurrent access is required.
// ---------------------------------------------------------------------------

class LockFreePoolAllocator {
public:
    static constexpr size_t DEFAULT_CAPACITY = 1u << 16;   // 65 536 orders

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit LockFreePoolAllocator(size_t capacity = DEFAULT_CAPACITY)
        : capacity_(capacity)
    {
        const size_t bytes = capacity_ * sizeof(Order);
        void* raw = std::aligned_alloc(alignof(Order), bytes);
        if (!raw) throw std::bad_alloc{};
        storage_.reset(static_cast<uint8_t*>(raw));

        // Thread the free list through every slot, back-to-front so that
        // the first allocation returns slot[0].
        TaggedPtr head{nullptr, 0};
        for (size_t i = capacity_; i-- > 0; ) {
            auto* node = reinterpret_cast<FreeNode*>(
                storage_.get() + i * sizeof(Order));
            node->next = head.ptr;
            head.ptr   = node;
            // No need to increment tag during construction — no concurrency yet.
        }

        // Store the initial head.  Plain store is fine here; no other
        // thread can be accessing the allocator during construction.
        free_head_.store(head, std::memory_order_relaxed);

        // Report whether we got a truly lock-free atomic on this platform.
        //
        // x86-64 (GCC -mcx16 / Clang default): is_lock_free() == true
        //   → uses CMPXCHG16B, single hardware instruction, no library lock.
        //
        // ARM64 with GCC + libatomic: is_lock_free() == false
        //   → uses a library-level lock internally; still ABA-safe and
        //     correct, but not a true hardware CAS.  Clang on ARM64 (e.g.
        //     Apple Silicon with Xcode toolchain) typically returns true.
        //
        // This distinction only affects performance, not correctness.
        if (!free_head_.is_lock_free()) {
            // In a real trading system you would abort here.  For this
            // educational project we proceed with a diagnostic message.
            __builtin_printf("[LockFreePoolAllocator] WARNING: "
                             "128-bit atomic is NOT hardware lock-free on "
                             "this platform.  Falling back to libatomic.\n"
                             "  On x86-64: compile with -mcx16.\n"
                             "  On ARM64:  use Clang (Apple toolchain).\n");
        }
    }

    ~LockFreePoolAllocator() = default;

    LockFreePoolAllocator(const LockFreePoolAllocator&)            = delete;
    LockFreePoolAllocator& operator=(const LockFreePoolAllocator&) = delete;
    LockFreePoolAllocator(LockFreePoolAllocator&&)                 = delete;
    LockFreePoolAllocator& operator=(LockFreePoolAllocator&&)      = delete;

    // -----------------------------------------------------------------------
    // allocate() — lock-free pop from the head of the free list
    //
    // Steps:
    //   1. Load the current head (pointer + tag) with acquire ordering.
    //   2. If null, pool is exhausted — return nullptr.
    //   3. Build the proposed new head: (head->next, same tag).
    //      We don't increment the tag on pop — only pushes increment it.
    //   4. CAS: if head still equals what we loaded, swap in new head.
    //      On success (acquire): return the old head slot as an Order*.
    //      On failure: another thread modified head; retry from step 1.
    //
    // Note on compare_exchange_weak vs strong:
    //   _weak may fail spuriously (without another thread interfering) on
    //   some architectures (LL/SC machines like ARM).  We use _weak here
    //   because we're already in a retry loop; a spurious failure just
    //   causes one extra iteration, which is cheaper than the extra
    //   guarantee _strong provides.  On x86-64, both compile to CMPXCHG16B
    //   and the difference is moot.
    // -----------------------------------------------------------------------
    [[nodiscard]] Order* allocate() noexcept {
        TaggedPtr current = free_head_.load(std::memory_order_acquire);

        while (true) {
            if (current.ptr == nullptr) [[unlikely]]
                return nullptr;   // pool exhausted

            // Read next *before* the CAS.  If the CAS succeeds, current.ptr
            // is ours and no other thread can modify its next pointer.
            TaggedPtr next_head{current.ptr->next, current.tag};

            // CAS: attempt to swing head from current to next_head.
            // Success ordering: acquire — synchronises with the release in
            //   deallocate() that pushed this node, ensuring we see the
            //   correct node->next value.
            // Failure ordering: acquire — ensures we re-read head
            //   consistently on the next iteration.
            if (free_head_.compare_exchange_weak(
                    current, next_head,
                    std::memory_order_acquire,
                    std::memory_order_acquire))
            {
                return reinterpret_cast<Order*>(current.ptr);
            }
            // current was updated with the true current value — retry.
        }
    }

    // -----------------------------------------------------------------------
    // deallocate() — lock-free push onto the head of the free list
    //
    // Steps:
    //   1. Load the current head with relaxed ordering (we only need the
    //      pointer value for the next-link; the acquire in allocate()
    //      provides the necessary synchronisation on the consuming side).
    //   2. Set node->next = current.ptr  (link into list).
    //   3. Build the proposed new head: (node, current.tag + 1).
    //      Incrementing the tag here is what prevents the ABA problem.
    //   4. CAS: if head still equals current, swap in new head.
    //      On success (release): our write to node->next becomes visible
    //        to any thread that subsequently pops this node.
    //      On failure: another thread modified head; retry from step 1.
    // -----------------------------------------------------------------------
    void deallocate(Order* ptr) noexcept {
        assert(ptr != nullptr);
        assert(owns(ptr) && "deallocate: pointer not from this pool");

        auto* node = reinterpret_cast<FreeNode*>(ptr);

        TaggedPtr current = free_head_.load(std::memory_order_relaxed);

        while (true) {
            node->next = current.ptr;

            // Increment the tag on every push to solve ABA.
            TaggedPtr new_head{node, current.tag + 1};

            // Success ordering: release — makes our write to node->next
            //   visible to the thread that eventually pops this node.
            // Failure ordering: relaxed — we're retrying anyway; the next
            //   iteration's load of current gives us the fresh value.
            if (free_head_.compare_exchange_weak(
                    current, new_head,
                    std::memory_order_release,
                    std::memory_order_relaxed))
            {
                return;
            }
            // current was updated with the true current value — retry.
        }
    }

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------

    size_t capacity() const noexcept { return capacity_; }

    bool owns(const Order* ptr) const noexcept {
        const auto* p   = reinterpret_cast<const uint8_t*>(ptr);
        const auto* beg = storage_.get();
        const auto* end = beg + capacity_ * sizeof(Order);
        return p >= beg && p < end;
    }

private:
    // -----------------------------------------------------------------------
    // Internal types
    // -----------------------------------------------------------------------

    // A free slot, viewed from the allocator's perspective.
    struct FreeNode {
        FreeNode* next;
    };
    static_assert(sizeof(FreeNode) <= sizeof(Order));

    // The 128-bit atomic head: pointer + version counter.
    //
    // alignas(16) is required in two places:
    //   - On the struct, so that sizeof == 16 and alignof == 16.
    //   - On the atomic member, so that its address is 16-byte aligned,
    //     which is required by CMPXCHG16B.
    //
    // Without alignas(16) on the member, the atomic may land at an address
    // that is only 8-byte aligned, causing CMPXCHG16B to fault at runtime
    // (#GP exception) or silently fall back to a mutex.
    struct alignas(16) TaggedPtr {
        FreeNode* ptr{nullptr};
        uintptr_t tag{0};
    };
    static_assert(sizeof(TaggedPtr)  == 16, "TaggedPtr must be 16 bytes for CMPXCHG16B");
    static_assert(alignof(TaggedPtr) == 16, "TaggedPtr must be 16-byte aligned");

    struct FreeDeleter {
        void operator()(uint8_t* p) const noexcept { std::free(p); }
    };

    std::unique_ptr<uint8_t[], FreeDeleter> storage_;

    // Place the atomic head on its own cache line.  This prevents false
    // sharing between the head and the storage array, and also ensures
    // the 16-byte alignment required by CMPXCHG16B (64 is a multiple of 16).
    alignas(64) std::atomic<TaggedPtr> free_head_{};

    size_t capacity_{0};
};
