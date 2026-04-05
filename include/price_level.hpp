#pragma once

#include <cstdint>
#include "order.hpp"

// ---------------------------------------------------------------------------
// PriceLevel
//
// Represents all resting orders at a single price tick on one side of the
// book.  Orders are kept in a FIFO queue (time-priority), implemented as an
// intrusive doubly-linked list using the Order::prev / Order::next pointers.
//
// Design notes:
//
//  - alignas(64): each PriceLevel occupies exactly one cache line.  The
//    array of PriceLevels in OrderBook is therefore accessed with zero
//    false-sharing — reading level[i] never pulls level[i+1] into the same
//    cache line.
//
//  - total_qty is maintained incrementally on every add/cancel/fill so the
//    strategy layer can read depth without traversing the linked list.
//
//  - head is the oldest (first-to-match) order; tail is the newest.
//    New orders are appended at tail; matching consumes from head.
//
//  - The struct is intentionally kept small so the hot fields (head, tail,
//    total_qty) fit in the first 24 bytes, maximising the chance that a
//    single cache line fetch contains everything the match engine needs.
// ---------------------------------------------------------------------------
struct alignas(64) PriceLevel {
    Order*   head;          // 8  — oldest resting order (matched first)
    Order*   tail;          // 8  — newest resting order (appended here)
    int64_t  price;         // 8  — price tick this level represents
    uint64_t total_qty;     // 8  — sum of leaves_qty() across all orders
    uint32_t order_count;   // 4  — number of live orders
    uint8_t  _pad[28];      // 28 — pad to 64 bytes

    // Returns true if there are no resting orders at this level
    [[nodiscard]] bool empty() const noexcept { return head == nullptr; }
};

static_assert(sizeof(PriceLevel)  == 64, "PriceLevel must be exactly one cache line");
static_assert(alignof(PriceLevel) == 64, "PriceLevel must be cache-line aligned");
