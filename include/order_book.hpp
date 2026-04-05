#pragma once

#include <array>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <optional>

#include "order.hpp"
#include "price_level.hpp"
#include "pool_allocator.hpp"

// ---------------------------------------------------------------------------
// OrderBook
//
// A price-time-priority central limit order book (CLOB) for a single
// instrument, supporting add, cancel, modify, and matching.
//
// Price representation:
//   All prices are integer ticks in the range [MIN_PRICE_TICK, MAX_PRICE_TICK].
//   The caller is responsible for converting from decimal prices to ticks
//   (e.g. price_in_ticks = static_cast<int64_t>(price * 100) for 2 d.p.).
//
// Data layout:
//   Two fixed arrays of PriceLevel (one for bids, one for asks) are indexed
//   directly by price tick.  This gives O(1) level lookup with a single array
//   dereference and no pointer chasing — far better cache behaviour than a
//   sorted map.  The cost is pre-allocating memory for the full price range
//   even if most levels are empty; for a 100 000-tick range that is
//   2 × 100 001 × 64 bytes ≈ 12.2 MiB, which is acceptable for a dedicated
//   trading process.
//
//   best_bid_ / best_ask_ are maintained incrementally: they are updated on
//   every add/cancel so that reading the BBO is a simple array index load
//   with no scan required.
//
// Order lookup:
//   order_map_ (unordered_map<id, Order*>) provides O(1) amortised cancel
//   and modify.  This map is *not* on the latency-critical add/match path;
//   it is only accessed for cancel and modify, which are less frequent.
//
// Memory:
//   All Order objects are allocated from a PoolAllocator (no malloc on the
//   hot path).  The pool is owned by the OrderBook and sized at construction.
//
// Thread safety:
//   NONE.  OrderBook is single-threaded by design.  Use SPSCQueue to feed
//   OrderEvents from a network-receive thread to the book thread.
// ---------------------------------------------------------------------------

// Sentinel values for best_bid_ / best_ask_ when the book is empty.
inline constexpr int64_t INVALID_PRICE = -1;

// ---------------------------------------------------------------------------
// Trade — emitted by the match engine whenever two orders cross
// ---------------------------------------------------------------------------
struct Trade {
    int64_t  price;               // execution price (ticks)
    uint32_t quantity;            // filled quantity
    uint64_t aggressive_order_id; // the incoming order that triggered the fill
    uint64_t passive_order_id;    // the resting order that was matched against
};

// ---------------------------------------------------------------------------
// OrderBook
// ---------------------------------------------------------------------------
class OrderBook {
public:
    // Price range covered by the pre-allocated level arrays.
    // Ticks outside this range are rejected by add_order().
    static constexpr int64_t MIN_PRICE_TICK = 0;
    static constexpr int64_t MAX_PRICE_TICK = 99'999;
    static constexpr size_t  NUM_LEVELS     = static_cast<size_t>(MAX_PRICE_TICK - MIN_PRICE_TICK + 1);

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    // pool_capacity: maximum number of simultaneously live Order objects.
    explicit OrderBook(size_t pool_capacity = PoolAllocator::DEFAULT_CAPACITY);

    ~OrderBook();

    // Non-copyable (owns raw memory via the pool)
    OrderBook(const OrderBook&)            = delete;
    OrderBook& operator=(const OrderBook&) = delete;

    // -----------------------------------------------------------------------
    // Mutating operations
    // -----------------------------------------------------------------------

    // Add a new resting order.  If the new order crosses the opposite side,
    // it is matched immediately (aggressor semantics) and any residual rests.
    // Returns false if price is out of range, qty is zero, or the pool is full.
    // Trades resulting from the match are appended to trades_out.
    bool add_order(uint64_t   order_id,
                   Side       side,
                   int64_t    price,
                   uint32_t   qty,
                   std::vector<Trade>& trades_out);

    // Cancel a resting order by id.  Returns false if order_id is unknown.
    bool cancel_order(uint64_t order_id);

    // Reduce the quantity of a resting order.  Only reductions are supported
    // (increasing qty would change time-priority).  Returns false if the order
    // is not found or new_qty >= current leaves_qty.
    bool modify_order(uint64_t order_id, uint32_t new_qty);

    // -----------------------------------------------------------------------
    // Accessors (read-only, lock-free)
    // -----------------------------------------------------------------------

    // Best bid price, or INVALID_PRICE if the bid side is empty.
    int64_t best_bid() const noexcept { return best_bid_; }

    // Best ask price, or INVALID_PRICE if the ask side is empty.
    int64_t best_ask() const noexcept { return best_ask_; }

    // Total quantity resting at a price level (0 if level is empty/invalid).
    uint64_t bid_qty_at(int64_t price) const noexcept;
    uint64_t ask_qty_at(int64_t price) const noexcept;

    // Number of live orders currently in the book.
    size_t order_count() const noexcept { return order_map_.size(); }

    // Allocator diagnostics.
    size_t pool_available() const noexcept { return pool_.available(); }

private:
    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    // Append order to the tail of the given price level's queue.
    void enqueue(PriceLevel& level, Order* o) noexcept;

    // Remove order from its price level queue (O(1) via prev/next).
    void dequeue(PriceLevel& level, Order* o) noexcept;

    // Execute a fill between aggressor and passive orders.
    // Returns the filled quantity.
    uint32_t fill(Order* aggressor, Order* passive,
                  std::vector<Trade>& trades_out) noexcept;

    // Match an incoming aggressor order against the opposite side.
    void match(Order* aggressor, std::vector<Trade>& trades_out);

    // Scan downward from current best_bid_ to find the new best.
    void update_best_bid_after_remove(int64_t removed_price) noexcept;
    // Scan upward from current best_ask_ to find the new best.
    void update_best_ask_after_remove(int64_t removed_price) noexcept;

    // Validate a price tick is in [MIN_PRICE_TICK, MAX_PRICE_TICK].
    static bool valid_price(int64_t price) noexcept {
        return price >= MIN_PRICE_TICK && price <= MAX_PRICE_TICK;
    }

    // Index into the level arrays from a price tick.
    static size_t level_index(int64_t price) noexcept {
        return static_cast<size_t>(price - MIN_PRICE_TICK);
    }

    // -----------------------------------------------------------------------
    // Data members
    // -----------------------------------------------------------------------

    // Heap-allocated arrays of PriceLevels (too large for the stack).
    // Using unique_ptr<T[]> avoids the 12 MiB stack frame.
    std::unique_ptr<PriceLevel[]> bids_;   // indexed by price tick
    std::unique_ptr<PriceLevel[]> asks_;   // indexed by price tick

    // O(1) order lookup for cancel / modify.
    std::unordered_map<uint64_t, Order*> order_map_;

    // Pool allocator — all Order objects come from here.
    PoolAllocator pool_;

    // Best bid (highest price with resting quantity) and best ask (lowest).
    // INVALID_PRICE when the respective side is empty.
    int64_t best_bid_{INVALID_PRICE};
    int64_t best_ask_{INVALID_PRICE};
};
