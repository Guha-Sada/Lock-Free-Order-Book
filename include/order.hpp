#pragma once

#include <cstdint>
#include <cstddef>

// ---------------------------------------------------------------------------
// Side
// ---------------------------------------------------------------------------
enum class Side : uint8_t {
    Bid = 0,   // buy side
    Ask = 1,   // sell side
};

// ---------------------------------------------------------------------------
// Order
//
// Design notes:
//
//  - Prices are stored as integer ticks (never floating-point).  This
//    eliminates all floating-point comparison hazards and maps directly to
//    how exchanges encode prices on the wire.
//
//  - The struct is padded to exactly 64 bytes (one x86-64 cache line) with
//    alignas(64).  This means no two adjacent orders share a cache line,
//    which prevents false-sharing when the pool allocator hands out
//    neighbouring slots to different threads.
//
//  - prev/next make Order an intrusive doubly-linked list node.  There is
//    no separate list-node allocation; the order *is* the node.  This halves
//    the number of heap allocations compared to a std::list<Order*> approach
//    and keeps the pointer chain inside the same pool memory region.
//
//  - filled_qty tracks partial fills so the match engine can compute the
//    residual without a separate lookup.
//
//  - The static_assert below is your canary: if you add a field and the
//    layout silently changes size, the build breaks immediately.
// ---------------------------------------------------------------------------
struct alignas(64) Order {
    uint64_t order_id;      // 8  — unique identifier assigned by caller
    uint64_t timestamp_ns;  // 8  — arrival time (CLOCK_MONOTONIC_RAW)
    int64_t  price;         // 8  — price in ticks (signed: allows negative spreads)
    uint32_t quantity;      // 4  — original order size
    uint32_t filled_qty;    // 4  — cumulative filled quantity
    Side     side;          // 1  — Bid or Ask
    uint8_t  _pad[7];       // 7  — explicit padding to align pointers
    Order*   prev;          // 8  — previous order at this price level (older)
    Order*   next;          // 8  — next order at this price level (newer)
    uint8_t  _reserved[8];  // 8  — reserved for future use (e.g. flags, type)
                            //      keeps sizeof == 64 after adding fields
};

static_assert(sizeof(Order)  == 64, "Order must be exactly one cache line (64 bytes)");
static_assert(alignof(Order) == 64, "Order must be cache-line aligned");

// Remaining quantity available to be filled
inline uint32_t leaves_qty(const Order& o) noexcept {
    return o.quantity - o.filled_qty;
}
