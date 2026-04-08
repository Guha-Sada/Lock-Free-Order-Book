#include "order_book.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <new>

// ---------------------------------------------------------------------------
// Helper: round up to the next power of two.
// Used when sizing the FlatHashMap: we want 4× the pool capacity, rounded up
// to a power of two (required by FlatHashMap's assertion).
// ---------------------------------------------------------------------------
static constexpr size_t next_pow2(size_t n) noexcept {
    if (n == 0) return 1;
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

OrderBook::OrderBook(size_t pool_capacity)
    : bids_(std::make_unique<PriceLevel[]>(NUM_LEVELS))
    , asks_(std::make_unique<PriceLevel[]>(NUM_LEVELS))
    // FlatHashMap capacity: 4× pool so load factor stays below 0.25 even
    // with tombstone accumulation.  Sized as a power of two (required for
    // the bitwise-AND modulo trick used by hash()).
    , order_map_(next_pow2(pool_capacity * 4))
    , pool_(pool_capacity)
{
    // Initialise every PriceLevel: make_unique value-initialises, but we
    // also explicitly set the price field so level_index lookups are correct.
    for (size_t i = 0; i < NUM_LEVELS; ++i) {
        bids_[i] = PriceLevel{};
        asks_[i] = PriceLevel{};
        bids_[i].price = static_cast<int64_t>(i) + MIN_PRICE_TICK;
        asks_[i].price = static_cast<int64_t>(i) + MIN_PRICE_TICK;
    }

    // No reserve() needed — FlatHashMap is sized upfront in the constructor.
}

OrderBook::~OrderBook() {
    // Order is trivially destructible (no owned resources).
    // pool_ owns all Order memory via its aligned_alloc block; when pool_'s
    // destructor runs it calls std::free() on the whole block in one shot.
    // No per-order cleanup is needed.
    //
    // order_map_ stores Order* pointers but does NOT own them — pool_ does.
    // FlatHashMap's destructor (delete[] slots_) frees the slot array only.
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

void OrderBook::enqueue(PriceLevel& level, Order* o) noexcept {
    o->prev = level.tail;
    o->next = nullptr;

    if (level.tail)
        level.tail->next = o;
    else
        level.head = o;   // first order at this level

    level.tail = o;
    level.total_qty += leaves_qty(*o);
    ++level.order_count;
}

void OrderBook::dequeue(PriceLevel& level, Order* o) noexcept {
    assert(level.order_count > 0);

    if (o->prev) o->prev->next = o->next;
    else         level.head    = o->next;   // o was the head

    if (o->next) o->next->prev = o->prev;
    else         level.tail    = o->prev;   // o was the tail

    o->prev = nullptr;
    o->next = nullptr;

    assert(level.total_qty >= leaves_qty(*o));
    level.total_qty -= leaves_qty(*o);
    --level.order_count;
}

uint32_t OrderBook::fill(Order* aggressor, Order* passive,
                         std::vector<Trade>& trades_out) noexcept {
    const uint32_t qty = std::min(leaves_qty(*aggressor), leaves_qty(*passive));
    assert(qty > 0);

    aggressor->filled_qty += qty;
    passive->filled_qty   += qty;

    trades_out.push_back(Trade{
        .price               = passive->price,
        .quantity            = qty,
        .aggressive_order_id = aggressor->order_id,
        .passive_order_id    = passive->order_id,
    });

    return qty;
}

void OrderBook::match(Order* aggressor, std::vector<Trade>& trades_out) {
    const bool is_bid = (aggressor->side == Side::Bid);

    while (leaves_qty(*aggressor) > 0) {
        // No liquidity on the opposite side
        if (is_bid  && best_ask_ == INVALID_PRICE) break;
        if (!is_bid && best_bid_ == INVALID_PRICE) break;

        // Price check: bid must be >= best ask to cross; ask must be <= best bid
        if (is_bid  && aggressor->price < best_ask_) break;
        if (!is_bid && aggressor->price > best_bid_) break;

        const int64_t match_price = is_bid ? best_ask_ : best_bid_;
        PriceLevel& level = is_bid
            ? asks_[level_index(match_price)]
            : bids_[level_index(match_price)];

        assert(!level.empty());
        Order* passive = level.head;

        const uint32_t filled = fill(aggressor, passive, trades_out);

        // Update level's aggregate quantity directly (avoids recomputing total).
        assert(level.total_qty >= filled);
        level.total_qty -= filled;

        if (leaves_qty(*passive) == 0) {
            // Passive order fully filled — remove from level and free.
            dequeue(level, passive);
            order_map_.erase(passive->order_id);
            passive->~Order();
            pool_.deallocate(passive);

            if (level.empty()) {
                if (is_bid)  update_best_ask_after_remove(match_price);
                else         update_best_bid_after_remove(match_price);
            }
        }
        // Partially filled passive stays at the head (time-priority preserved).
    }
}

void OrderBook::update_best_bid_after_remove(int64_t removed_price) noexcept {
    if (removed_price != best_bid_) return;
    for (int64_t p = best_bid_ - 1; p >= MIN_PRICE_TICK; --p) {
        if (!bids_[level_index(p)].empty()) {
            best_bid_ = p;
            return;
        }
    }
    best_bid_ = INVALID_PRICE;
}

void OrderBook::update_best_ask_after_remove(int64_t removed_price) noexcept {
    if (removed_price != best_ask_) return;
    for (int64_t p = best_ask_ + 1; p <= MAX_PRICE_TICK; ++p) {
        if (!asks_[level_index(p)].empty()) {
            best_ask_ = p;
            return;
        }
    }
    best_ask_ = INVALID_PRICE;
}

// ---------------------------------------------------------------------------
// Public: add_order
// ---------------------------------------------------------------------------

bool OrderBook::add_order(uint64_t order_id, Side side, int64_t price,
                          uint32_t qty, std::vector<Trade>& trades_out) {
    if (!valid_price(price)) return false;
    if (qty == 0)            return false;

    // Duplicate check: FlatHashMap::find returns nullptr if not present.
    if (order_map_.find(order_id) != nullptr) return false;

    Order* o = pool_.allocate();
    if (!o) [[unlikely]] return false;  // pool exhausted

    new (o) Order{
        .order_id     = order_id,
        .timestamp_ns = 0,
        .price        = price,
        .quantity     = qty,
        .filled_qty   = 0,
        .side         = side,
        ._pad         = {},
        .prev         = nullptr,
        .next         = nullptr,
        ._reserved    = {},
    };

    // FlatHashMap::insert(key, value) — O(1) amortised, no heap allocation.
    order_map_.insert(order_id, o);

    match(o, trades_out);

    if (leaves_qty(*o) > 0) {
        PriceLevel& level = (side == Side::Bid)
            ? bids_[level_index(price)]
            : asks_[level_index(price)];

        enqueue(level, o);

        if (side == Side::Bid) {
            if (best_bid_ == INVALID_PRICE || price > best_bid_)
                best_bid_ = price;
        } else {
            if (best_ask_ == INVALID_PRICE || price < best_ask_)
                best_ask_ = price;
        }
    } else {
        // Fully matched — no resting quantity.
        order_map_.erase(order_id);
        o->~Order();
        pool_.deallocate(o);
    }

    return true;
}

// ---------------------------------------------------------------------------
// Public: cancel_order
// ---------------------------------------------------------------------------

bool OrderBook::cancel_order(uint64_t order_id) {
    // find() returns Order** (pointer to the value slot), or nullptr.
    Order** slot = order_map_.find(order_id);
    if (!slot) return false;

    Order* o = *slot;
    order_map_.erase(order_id);

    PriceLevel& level = (o->side == Side::Bid)
        ? bids_[level_index(o->price)]
        : asks_[level_index(o->price)];

    dequeue(level, o);

    if (level.empty()) {
        if (o->side == Side::Bid) update_best_bid_after_remove(o->price);
        else                      update_best_ask_after_remove(o->price);
    }

    o->~Order();
    pool_.deallocate(o);
    return true;
}

// ---------------------------------------------------------------------------
// Public: modify_order
// ---------------------------------------------------------------------------

bool OrderBook::modify_order(uint64_t order_id, uint32_t new_qty) {
    Order** slot = order_map_.find(order_id);
    if (!slot) return false;

    Order* o = *slot;
    const uint32_t current_leaves = leaves_qty(*o);

    // Only reductions: increasing qty would require re-prioritising.
    if (new_qty == 0 || new_qty >= current_leaves) return false;

    PriceLevel& level = (o->side == Side::Bid)
        ? bids_[level_index(o->price)]
        : asks_[level_index(o->price)];

    const uint32_t reduction = current_leaves - new_qty;
    assert(level.total_qty >= reduction);
    level.total_qty -= reduction;

    o->quantity = o->filled_qty + new_qty;
    return true;
}

// ---------------------------------------------------------------------------
// Public: accessors
// ---------------------------------------------------------------------------

uint64_t OrderBook::bid_qty_at(int64_t price) const noexcept {
    if (!valid_price(price)) return 0;
    return bids_[level_index(price)].total_qty;
}

uint64_t OrderBook::ask_qty_at(int64_t price) const noexcept {
    if (!valid_price(price)) return 0;
    return asks_[level_index(price)].total_qty;
}
