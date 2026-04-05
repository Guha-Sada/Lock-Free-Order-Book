#pragma once

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <memory>
#include <new>
#include <stdexcept>

#include "order.hpp"

// ---------------------------------------------------------------------------
// PoolAllocator  (Version A — single-threaded)
//
// A fixed-capacity slab allocator for Order objects.  All memory is acquired
// once at construction time via aligned_alloc; every subsequent allocate()
// and deallocate() is O(1) with zero system calls.
//
// How it works:
//
//   1. At construction, the storage block is carved into N Order-sized slots.
//   2. A singly-linked free list is threaded *through the free slots
//      themselves*: the first bytes of each free slot store a FreeNode
//      pointer to the next free slot.  No separate free-list array is needed.
//   3. allocate()   — pop the head of the free list, return that slot.
//   4. deallocate() — push the slot back onto the head of the free list.
//
// Both operations are a single pointer swap: O(1), branch-free, no malloc.
//
// Thread safety:
//   NONE.  This allocator is designed to be owned and used exclusively by
//   the order-book thread.  See the comment at the bottom of this file for
//   the lock-free Version B design (ABA-safe atomic free list) that you will
//   build in Phase 3.
//
// Why not std::pmr::pool_resource?
//   The PMR pool has good average-case performance but its internal bookkeep-
//   ing can cause cache misses and its interface imposes virtual dispatch.
//   We want a zero-overhead, inlineable, domain-specific allocator.
// ---------------------------------------------------------------------------

class PoolAllocator {
public:
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    // capacity: maximum number of Order objects the pool can hold at once.
    // The pool allocates capacity * sizeof(Order) bytes upfront.
    // Default: 65 536 orders → 4 MiB (comfortably fits in LLC on modern CPUs)
    static constexpr size_t DEFAULT_CAPACITY = 1u << 16;  // 65 536

    explicit PoolAllocator(size_t capacity = DEFAULT_CAPACITY)
        : capacity_(capacity)
    {
        // aligned_alloc requires size to be a multiple of alignment.
        const size_t bytes = capacity_ * sizeof(Order);

        // Use aligned_alloc so the storage block begins on a 64-byte boundary.
        // This ensures that slot[0] satisfies Order's alignas(64) requirement,
        // and because sizeof(Order)==64, every subsequent slot is also aligned.
        void* raw = std::aligned_alloc(alignof(Order), bytes);
        if (!raw)
            throw std::bad_alloc{};

        storage_.reset(static_cast<uint8_t*>(raw));

        // Thread the free list through every slot.  We walk backwards so
        // that the first allocation returns slot[0] (lowest address first).
        free_head_ = nullptr;
        for (size_t i = capacity_; i-- > 0; ) {
            auto* node = reinterpret_cast<FreeNode*>(storage_.get() + i * sizeof(Order));
            node->next = free_head_;
            free_head_ = node;
        }
        available_ = capacity_;
    }

    ~PoolAllocator() = default;

    // Non-copyable, non-movable (raw pointer math breaks on move)
    PoolAllocator(const PoolAllocator&)            = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;
    PoolAllocator(PoolAllocator&&)                 = delete;
    PoolAllocator& operator=(PoolAllocator&&)      = delete;

    // -----------------------------------------------------------------------
    // Core operations
    // -----------------------------------------------------------------------

    // Returns a pointer to an uninitialised Order-sized slot.
    // The caller is responsible for placement-new to construct the Order.
    // Returns nullptr if the pool is exhausted.
    [[nodiscard]] Order* allocate() noexcept {
        if (!free_head_) [[unlikely]]
            return nullptr;

        FreeNode* node = free_head_;
        free_head_ = node->next;
        --available_;

        // The slot's bytes are uninitialised — exactly like malloc.
        // The caller must construct an Order via placement-new.
        return reinterpret_cast<Order*>(node);
    }

    // Returns an Order slot back to the pool.
    // The caller must have already destroyed (not deleted) the Order.
    // UB if ptr was not obtained from this pool instance.
    void deallocate(Order* ptr) noexcept {
        assert(ptr != nullptr);
        assert(owns(ptr) && "deallocate: pointer not from this pool");

        // Reuse the slot's bytes to store the free-list linkage.
        auto* node = reinterpret_cast<FreeNode*>(ptr);
        node->next = free_head_;
        free_head_ = node;
        ++available_;
    }

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------
    size_t available() const noexcept { return available_; }
    size_t capacity()  const noexcept { return capacity_;  }
    size_t in_use()    const noexcept { return capacity_ - available_; }

    // Returns true if ptr points into this pool's storage block.
    bool owns(const Order* ptr) const noexcept {
        const auto* p   = reinterpret_cast<const uint8_t*>(ptr);
        const auto* beg = storage_.get();
        const auto* end = beg + capacity_ * sizeof(Order);
        return p >= beg && p < end;
    }

private:
    // A free slot looks like this from the allocator's perspective.
    // sizeof(FreeNode) must fit inside sizeof(Order).
    struct FreeNode {
        FreeNode* next;
    };
    static_assert(sizeof(FreeNode) <= sizeof(Order),
                  "Order is too small to hold a FreeNode — increase Order size");

    // Custom deleter calls free() instead of delete[] (we used aligned_alloc)
    struct FreeDeleter {
        void operator()(uint8_t* p) const noexcept { std::free(p); }
    };

    std::unique_ptr<uint8_t[], FreeDeleter> storage_;
    FreeNode* free_head_{nullptr};
    size_t    capacity_{0};
    size_t    available_{0};
};

// ---------------------------------------------------------------------------
// NOTE — Version B (lock-free, Phase 3):
//
// Replace free_head_ with:
//     struct TaggedPtr { FreeNode* ptr; uintptr_t tag; };
//     std::atomic<TaggedPtr> free_head_;
//
// The 'tag' counter increments on every push, solving the ABA problem:
// if thread A pops node X, thread B pops X then pushes X back, thread A's
// CAS on (X, old_tag) will fail because the tag has changed, forcing a retry.
//
// On x86-64, a 128-bit compare-and-swap (CMPXCHG16B) can atomically swap
// both the pointer and its tag in one instruction.  Use __int128 or a
// std::pair with std::atomic<> (requires -mcx16 on GCC).
// ---------------------------------------------------------------------------
