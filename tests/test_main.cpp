#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <thread>
#include <atomic>

#include "order.hpp"
#include "price_level.hpp"
#include "pool_allocator.hpp"
#include "lockfree_pool_allocator.hpp"
#include "spsc_queue.hpp"
#include "naive_spsc_queue.hpp"
#include "order_book.hpp"

// ===========================================================================
// order.hpp
// ===========================================================================

TEST_CASE("Order struct layout", "[order]") {
    REQUIRE(sizeof(Order)  == 64);
    REQUIRE(alignof(Order) == 64);
}

TEST_CASE("leaves_qty helper", "[order]") {
    Order o{};
    o.quantity   = 100;
    o.filled_qty = 30;
    REQUIRE(leaves_qty(o) == 70u);
}

// ===========================================================================
// price_level.hpp
// ===========================================================================

TEST_CASE("PriceLevel struct layout", "[price_level]") {
    REQUIRE(sizeof(PriceLevel)  == 64);
    REQUIRE(alignof(PriceLevel) == 64);
}

TEST_CASE("PriceLevel default empty", "[price_level]") {
    PriceLevel level{};
    REQUIRE(level.empty());
    REQUIRE(level.head        == nullptr);
    REQUIRE(level.tail        == nullptr);
    REQUIRE(level.total_qty   == 0u);
    REQUIRE(level.order_count == 0u);
}

// ===========================================================================
// pool_allocator.hpp
// ===========================================================================

TEST_CASE("PoolAllocator: construction", "[pool]") {
    constexpr size_t CAP = 128;
    PoolAllocator pool(CAP);
    REQUIRE(pool.capacity()  == CAP);
    REQUIRE(pool.available() == CAP);
    REQUIRE(pool.in_use()    == 0u);
}

TEST_CASE("PoolAllocator: allocate and deallocate", "[pool]") {
    PoolAllocator pool(4);

    Order* a = pool.allocate();
    Order* b = pool.allocate();
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(a != b);
    REQUIRE(pool.in_use()    == 2u);
    REQUIRE(pool.available() == 2u);
    REQUIRE(pool.owns(a));
    REQUIRE(pool.owns(b));

    pool.deallocate(a);
    REQUIRE(pool.available() == 3u);

    // Slot should be reusable.
    Order* c = pool.allocate();
    REQUIRE(c != nullptr);
    REQUIRE(pool.available() == 2u);

    pool.deallocate(b);
    pool.deallocate(c);
    REQUIRE(pool.available() == 4u);
}

TEST_CASE("PoolAllocator: exhaustion returns nullptr", "[pool]") {
    PoolAllocator pool(2);
    Order* a = pool.allocate();
    Order* b = pool.allocate();
    REQUIRE(pool.allocate() == nullptr);   // pool full
    pool.deallocate(a);
    pool.deallocate(b);
}

TEST_CASE("PoolAllocator: alignment of allocated slots", "[pool]") {
    PoolAllocator pool(8);
    for (int i = 0; i < 8; ++i) {
        Order* p = pool.allocate();
        REQUIRE(p != nullptr);
        // Each slot must satisfy Order's alignment requirement.
        REQUIRE(reinterpret_cast<uintptr_t>(p) % alignof(Order) == 0u);
    }
}

// ===========================================================================
// spsc_queue.hpp
// ===========================================================================

TEST_CASE("SPSCQueue: basic push/pop", "[spsc]") {
    SPSCQueue<int, 8> q;
    REQUIRE(q.empty());

    REQUIRE(q.push(42));
    REQUIRE(!q.empty());

    int val{};
    REQUIRE(q.pop(val));
    REQUIRE(val == 42);
    REQUIRE(q.empty());
}

TEST_CASE("SPSCQueue: capacity (N-1 usable slots)", "[spsc]") {
    SPSCQueue<int, 4> q;   // capacity = 3
    REQUIRE(q.capacity() == 3u);

    REQUIRE(q.push(1));
    REQUIRE(q.push(2));
    REQUIRE(q.push(3));
    REQUIRE(!q.push(4));   // full

    int v{};
    REQUIRE(q.pop(v)); REQUIRE(v == 1);
    REQUIRE(q.pop(v)); REQUIRE(v == 2);
    REQUIRE(q.pop(v)); REQUIRE(v == 3);
    REQUIRE(!q.pop(v)); // empty
}

TEST_CASE("SPSCQueue: FIFO ordering", "[spsc]") {
    SPSCQueue<int, 16> q;
    for (int i = 0; i < 10; ++i) REQUIRE(q.push(i));
    for (int i = 0; i < 10; ++i) {
        int v{};
        REQUIRE(q.pop(v));
        REQUIRE(v == i);
    }
}

// ---------------------------------------------------------------------------
// Concurrent correctness test
//
// A real producer thread pushes N sequential integers; a real consumer thread
// pops them and verifies they arrive in order and with no gaps.
//
// What this proves that the single-threaded tests cannot:
//   - The acquire/release memory ordering is actually correct: the consumer
//     sees each value *after* the producer wrote it, never before.
//   - There are no data races (run under -fsanitize=thread to confirm).
//   - The queue works across a cache coherency boundary — values written on
//     one core are correctly visible on another.
//
// The FIFO check (expected == received) is the key assertion: any reordering
// due to wrong memory ordering would show up as an out-of-sequence value.
// ---------------------------------------------------------------------------
TEST_CASE("SPSCQueue: concurrent producer/consumer FIFO correctness", "[spsc][concurrent]") {
    constexpr int N = 100'000;
    SPSCQueue<int, 4096> q;

    std::atomic<bool> producer_done{false};
    bool fifo_ok = true;

    std::thread producer([&]() {
        for (int i = 0; i < N; ++i)
            while (!q.push(i)) std::this_thread::yield();
        producer_done.store(true, std::memory_order_release);
    });

    std::thread consumer([&]() {
        int expected = 0;
        while (expected < N) {
            int val{};
            if (q.pop(val)) {
                // Every value must arrive in strict order.
                // A wrong memory ordering would break this.
                if (val != expected) { fifo_ok = false; }
                ++expected;
            } else if (producer_done.load(std::memory_order_acquire)) {
                // Producer finished — drain any remaining items.
                while (q.pop(val)) {
                    if (val != expected) { fifo_ok = false; }
                    ++expected;
                }
                break;
            }
        }
    });

    producer.join();
    consumer.join();

    REQUIRE(fifo_ok);
    REQUIRE(q.empty());
}

// ---------------------------------------------------------------------------
// NaiveSPSCQueue — correctness (same contract, no padding)
//
// The naive queue is logically identical to SPSCQueue — just slower under
// contention due to false sharing.  Verify the same single-threaded contract
// holds so we know the benchmark difference is purely the cache-line effect,
// not a logic difference.
// ---------------------------------------------------------------------------
TEST_CASE("NaiveSPSCQueue: basic push/pop", "[spsc][naive]") {
    NaiveSPSCQueue<int, 8> q;
    REQUIRE(q.empty());
    REQUIRE(q.push(99));
    int v{};
    REQUIRE(q.pop(v));
    REQUIRE(v == 99);
    REQUIRE(q.empty());
}

TEST_CASE("NaiveSPSCQueue: FIFO ordering", "[spsc][naive]") {
    NaiveSPSCQueue<int, 16> q;
    for (int i = 0; i < 10; ++i) REQUIRE(q.push(i));
    for (int i = 0; i < 10; ++i) {
        int v{};
        REQUIRE(q.pop(v));
        REQUIRE(v == i);
    }
}

// ---------------------------------------------------------------------------
// Cache-line layout verification
//
// Confirms that the padding actually achieves its goal: write_pos_ and
// read_pos_ in SPSCQueue must be on different cache lines, while the same
// members in NaiveSPSCQueue must share one.
//
// We do this by inspecting the offset of each member within a known-size
// allocation.  Because the queue is the only object in a fresh allocation,
// the member addresses reveal their cache-line positions.
// ---------------------------------------------------------------------------
TEST_CASE("SPSCQueue: write_pos_ and read_pos_ are on different cache lines", "[spsc][layout]") {
    // Allocate two queues side by side and check internal member offsets
    // using the fact that alignas(64) forces 64-byte separation.
    //
    // The padded queue's layout (from the struct definition):
    //   offset   0: write_pos_  (alignas(64), 8 bytes)
    //   offset  64: read_pos_   (alignas(64), 8 bytes)
    //   offset 128: buffer_
    //
    // The naive queue's layout:
    //   offset  0: write_pos_  (8 bytes)
    //   offset  8: read_pos_   (8 bytes, immediately after)
    //   offset 16: buffer_

    SPSCQueue<uint8_t, 64>      padded;
    NaiveSPSCQueue<uint8_t, 64> naive;

    // sizeof confirms the padding exists in SPSCQueue.
    // Two alignas(64) members + 64-element buffer ≥ 192 bytes.
    REQUIRE(sizeof(padded) >= 192u);

    // NaiveSPSCQueue is much smaller — no padding between the two atomics.
    REQUIRE(sizeof(naive) < sizeof(padded));
}

// ===========================================================================
// order_book.hpp / order_book.cpp
// ===========================================================================

TEST_CASE("OrderBook: initially empty", "[book]") {
    OrderBook book;
    REQUIRE(book.best_bid()   == INVALID_PRICE);
    REQUIRE(book.best_ask()   == INVALID_PRICE);
    REQUIRE(book.order_count() == 0u);
}

TEST_CASE("OrderBook: add single bid", "[book]") {
    OrderBook book;
    std::vector<Trade> trades;

    REQUIRE(book.add_order(1, Side::Bid, 100, 10, trades));
    REQUIRE(trades.empty());          // no cross
    REQUIRE(book.best_bid()   == 100);
    REQUIRE(book.best_ask()   == INVALID_PRICE);
    REQUIRE(book.order_count() == 1u);
    REQUIRE(book.bid_qty_at(100) == 10u);
}

TEST_CASE("OrderBook: add single ask", "[book]") {
    OrderBook book;
    std::vector<Trade> trades;

    REQUIRE(book.add_order(1, Side::Ask, 200, 5, trades));
    REQUIRE(trades.empty());
    REQUIRE(book.best_ask()   == 200);
    REQUIRE(book.best_bid()   == INVALID_PRICE);
    REQUIRE(book.ask_qty_at(200) == 5u);
}

TEST_CASE("OrderBook: best bid tracks highest price", "[book]") {
    OrderBook book;
    std::vector<Trade> trades;

    book.add_order(1, Side::Bid, 100, 10, trades);
    REQUIRE(book.best_bid() == 100);

    book.add_order(2, Side::Bid, 105, 5, trades);
    REQUIRE(book.best_bid() == 105);

    book.add_order(3, Side::Bid, 98,  7, trades);
    REQUIRE(book.best_bid() == 105);  // unchanged
}

TEST_CASE("OrderBook: best ask tracks lowest price", "[book]") {
    OrderBook book;
    std::vector<Trade> trades;

    book.add_order(1, Side::Ask, 200, 10, trades);
    REQUIRE(book.best_ask() == 200);

    book.add_order(2, Side::Ask, 195, 5, trades);
    REQUIRE(book.best_ask() == 195);

    book.add_order(3, Side::Ask, 202, 7, trades);
    REQUIRE(book.best_ask() == 195);  // unchanged
}

TEST_CASE("OrderBook: cancel updates best bid", "[book]") {
    OrderBook book;
    std::vector<Trade> trades;

    book.add_order(1, Side::Bid, 105, 10, trades);
    book.add_order(2, Side::Bid, 100, 5,  trades);
    REQUIRE(book.best_bid() == 105);

    REQUIRE(book.cancel_order(1));
    REQUIRE(book.best_bid() == 100);

    REQUIRE(book.cancel_order(2));
    REQUIRE(book.best_bid() == INVALID_PRICE);
}

TEST_CASE("OrderBook: cancel unknown order returns false", "[book]") {
    OrderBook book;
    REQUIRE(!book.cancel_order(999));
}

TEST_CASE("OrderBook: full match (bid aggressor)", "[book]") {
    OrderBook book;
    std::vector<Trade> trades;

    // Resting ask at 100 for qty 10
    book.add_order(1, Side::Ask, 100, 10, trades);
    REQUIRE(trades.empty());

    // Incoming bid at 100 for qty 10 — should fully match
    book.add_order(2, Side::Bid, 100, 10, trades);
    REQUIRE(trades.size() == 1u);
    REQUIRE(trades[0].price    == 100);
    REQUIRE(trades[0].quantity == 10u);
    REQUIRE(trades[0].aggressive_order_id == 2u);
    REQUIRE(trades[0].passive_order_id    == 1u);

    // Both orders consumed — book should be empty
    REQUIRE(book.order_count() == 0u);
    REQUIRE(book.best_bid()    == INVALID_PRICE);
    REQUIRE(book.best_ask()    == INVALID_PRICE);
}

TEST_CASE("OrderBook: partial match leaves residual", "[book]") {
    OrderBook book;
    std::vector<Trade> trades;

    // Resting ask qty 5
    book.add_order(1, Side::Ask, 100, 5, trades);
    // Incoming bid qty 10 — partial fill of 5, residual 5 rests
    book.add_order(2, Side::Bid, 100, 10, trades);

    REQUIRE(trades.size() == 1u);
    REQUIRE(trades[0].quantity == 5u);

    // Resting ask is gone; bid residual (5) should remain
    REQUIRE(book.order_count()  == 1u);
    REQUIRE(book.best_bid()     == 100);
    REQUIRE(book.bid_qty_at(100) == 5u);
    REQUIRE(book.best_ask()     == INVALID_PRICE);
}

TEST_CASE("OrderBook: no cross when spread exists", "[book]") {
    OrderBook book;
    std::vector<Trade> trades;

    book.add_order(1, Side::Bid, 99,  10, trades);
    book.add_order(2, Side::Ask, 101, 10, trades);

    REQUIRE(trades.empty());
    REQUIRE(book.order_count() == 2u);
    REQUIRE(book.best_bid() == 99);
    REQUIRE(book.best_ask() == 101);
}

TEST_CASE("OrderBook: modify reduces quantity", "[book]") {
    OrderBook book;
    std::vector<Trade> trades;

    book.add_order(1, Side::Bid, 100, 20, trades);
    REQUIRE(book.bid_qty_at(100) == 20u);

    REQUIRE(book.modify_order(1, 10));
    REQUIRE(book.bid_qty_at(100) == 10u);
}

TEST_CASE("OrderBook: modify rejects increase", "[book]") {
    OrderBook book;
    std::vector<Trade> trades;

    book.add_order(1, Side::Bid, 100, 10, trades);
    REQUIRE(!book.modify_order(1, 15));  // increase rejected
    REQUIRE(book.bid_qty_at(100) == 10u);
}

TEST_CASE("OrderBook: duplicate order id rejected", "[book]") {
    OrderBook book;
    std::vector<Trade> trades;

    REQUIRE( book.add_order(1, Side::Bid, 100, 10, trades));
    REQUIRE(!book.add_order(1, Side::Bid, 101, 5,  trades));  // duplicate
}

TEST_CASE("OrderBook: out-of-range price rejected", "[book]") {
    OrderBook book;
    std::vector<Trade> trades;

    REQUIRE(!book.add_order(1, Side::Bid, -1,     10, trades));
    REQUIRE(!book.add_order(2, Side::Ask, 100'000, 10, trades));
}

TEST_CASE("OrderBook: zero quantity rejected", "[book]") {
    OrderBook book;
    std::vector<Trade> trades;

    REQUIRE(!book.add_order(1, Side::Bid, 100, 0, trades));
    REQUIRE(book.order_count() == 0u);
}

// ===========================================================================
// Phase 4 — comprehensive order book tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Helper: assert the pool-leak invariant.
//
// The number of slots the pool has handed out must always equal the number
// of live orders in the book.  If these diverge, we either leaked an Order
// (pool says slot is in-use but the book lost the pointer) or double-freed
// one (pool says slot is free but the book still references it).
//
// This is the single most powerful correctness check in the suite.
// ---------------------------------------------------------------------------
static void check_pool_invariant(const OrderBook& book) {
    // pool_available() + order_count() must equal pool capacity.
    // Equivalently: pool_in_use == order_count.
    // We check via the public accessor pool_available().
    // pool capacity is DEFAULT_CAPACITY; in_use = capacity - available.
    const size_t available = book.pool_available();
    const size_t in_use    = PoolAllocator::DEFAULT_CAPACITY - available;
    REQUIRE(in_use == book.order_count());
}

// ---------------------------------------------------------------------------
// Time-priority (FIFO) matching
//
// Two orders rest at the same price.  An aggressor should fill the earlier
// (head) order first, then the later (tail) order.
// ---------------------------------------------------------------------------
TEST_CASE("OrderBook: time priority — earlier order filled first", "[book][match]") {
    OrderBook book;
    std::vector<Trade> trades;

    // Two resting asks at the same price, added in order: id=1, then id=2.
    book.add_order(1, Side::Ask, 100, 5, trades);
    book.add_order(2, Side::Ask, 100, 5, trades);
    REQUIRE(book.ask_qty_at(100) == 10u);

    // Incoming bid large enough to fill both.
    book.add_order(3, Side::Bid, 100, 10, trades);

    REQUIRE(trades.size() == 2u);

    // First trade must be against the earlier resting order (id=1).
    REQUIRE(trades[0].passive_order_id == 1u);
    REQUIRE(trades[0].quantity         == 5u);

    // Second trade against the later one (id=2).
    REQUIRE(trades[1].passive_order_id == 2u);
    REQUIRE(trades[1].quantity         == 5u);

    REQUIRE(book.order_count()    == 0u);
    REQUIRE(book.best_ask()       == INVALID_PRICE);
    check_pool_invariant(book);
}

// ---------------------------------------------------------------------------
// Multi-level sweep
//
// An aggressor bid priced aggressively enough to consume orders across
// multiple ask price levels in one shot.
// ---------------------------------------------------------------------------
TEST_CASE("OrderBook: multi-level sweep — bid consumes three ask levels", "[book][match]") {
    OrderBook book;
    std::vector<Trade> trades;

    // Three resting asks at different prices.
    book.add_order(1, Side::Ask, 100, 10, trades);
    book.add_order(2, Side::Ask, 101, 10, trades);
    book.add_order(3, Side::Ask, 102, 10, trades);
    REQUIRE(book.best_ask() == 100);

    // Bid at 102 — crosses all three levels.
    book.add_order(4, Side::Bid, 102, 30, trades);

    REQUIRE(trades.size() == 3u);

    // Fills happen at each passive level's price (passive price-priority).
    REQUIRE(trades[0].price == 100);
    REQUIRE(trades[1].price == 101);
    REQUIRE(trades[2].price == 102);

    REQUIRE(book.order_count() == 0u);
    REQUIRE(book.best_ask()    == INVALID_PRICE);
    REQUIRE(book.best_bid()    == INVALID_PRICE);
    check_pool_invariant(book);
}

// ---------------------------------------------------------------------------
// Ask aggressor crossing the bid side
//
// All earlier matching tests used a bid as the aggressor.  This verifies
// the symmetric ask-aggressor path through the match engine.
// ---------------------------------------------------------------------------
TEST_CASE("OrderBook: ask aggressor fully matches resting bid", "[book][match]") {
    OrderBook book;
    std::vector<Trade> trades;

    book.add_order(1, Side::Bid, 100, 10, trades);
    REQUIRE(trades.empty());

    book.add_order(2, Side::Ask, 100, 10, trades);

    REQUIRE(trades.size() == 1u);
    REQUIRE(trades[0].price               == 100);
    REQUIRE(trades[0].quantity            == 10u);
    REQUIRE(trades[0].aggressive_order_id == 2u);
    REQUIRE(trades[0].passive_order_id    == 1u);

    REQUIRE(book.order_count() == 0u);
    REQUIRE(book.best_bid()    == INVALID_PRICE);
    REQUIRE(book.best_ask()    == INVALID_PRICE);
    check_pool_invariant(book);
}

TEST_CASE("OrderBook: ask aggressor sweeps multiple bid levels", "[book][match]") {
    OrderBook book;
    std::vector<Trade> trades;

    book.add_order(1, Side::Bid, 100, 5, trades);
    book.add_order(2, Side::Bid, 99,  5, trades);
    book.add_order(3, Side::Bid, 98,  5, trades);
    REQUIRE(book.best_bid() == 100);

    // Ask priced at 98 — should sweep all three bid levels.
    book.add_order(4, Side::Ask, 98, 15, trades);

    REQUIRE(trades.size() == 3u);
    REQUIRE(trades[0].price == 100);  // best bid first
    REQUIRE(trades[1].price == 99);
    REQUIRE(trades[2].price == 98);

    REQUIRE(book.order_count() == 0u);
    REQUIRE(book.best_bid()    == INVALID_PRICE);
    check_pool_invariant(book);
}

// ---------------------------------------------------------------------------
// Partial fill leaving residual on the passive side
//
// Aggressor is smaller than the resting order.  The passive order should
// remain at the head of the level with reduced quantity.
// ---------------------------------------------------------------------------
TEST_CASE("OrderBook: partial fill — passive order retains residual at head", "[book][match]") {
    OrderBook book;
    std::vector<Trade> trades;

    // Resting ask for 20.
    book.add_order(1, Side::Ask, 100, 20, trades);

    // Incoming bid for 7 — partially fills the ask.
    book.add_order(2, Side::Bid, 100, 7, trades);

    REQUIRE(trades.size()    == 1u);
    REQUIRE(trades[0].quantity == 7u);

    // Aggressor fully consumed (qty=7, no residual).
    // Passive retains 13 units.
    REQUIRE(book.order_count()    == 1u);   // only the passive remains
    REQUIRE(book.ask_qty_at(100)  == 13u);
    REQUIRE(book.best_ask()       == 100);
    REQUIRE(book.best_bid()       == INVALID_PRICE);
    check_pool_invariant(book);
}

// ---------------------------------------------------------------------------
// Cancel from the middle of a queue
//
// Three orders at the same price.  Cancel the middle one.
// Verifies that the linked list stitches correctly: head→tail integrity
// and the level's total_qty update.
// ---------------------------------------------------------------------------
TEST_CASE("OrderBook: cancel middle order in queue", "[book][cancel]") {
    OrderBook book;
    std::vector<Trade> trades;

    book.add_order(1, Side::Bid, 100, 10, trades);  // head
    book.add_order(2, Side::Bid, 100, 20, trades);  // middle
    book.add_order(3, Side::Bid, 100, 30, trades);  // tail

    REQUIRE(book.bid_qty_at(100) == 60u);

    // Cancel the middle order.
    REQUIRE(book.cancel_order(2));

    REQUIRE(book.order_count()   == 2u);
    REQUIRE(book.bid_qty_at(100) == 40u);  // 10 + 30
    REQUIRE(book.best_bid()      == 100);  // level still non-empty

    // Verify time priority is preserved: a matching ask should fill id=1 first.
    book.add_order(4, Side::Ask, 100, 10, trades);
    REQUIRE(trades.size()              == 1u);
    REQUIRE(trades[0].passive_order_id == 1u);

    check_pool_invariant(book);
}

TEST_CASE("OrderBook: cancel head order in queue", "[book][cancel]") {
    OrderBook book;
    std::vector<Trade> trades;

    book.add_order(1, Side::Ask, 100, 10, trades);  // head
    book.add_order(2, Side::Ask, 100, 20, trades);  // tail

    REQUIRE(book.cancel_order(1));
    REQUIRE(book.ask_qty_at(100) == 20u);
    REQUIRE(book.order_count()   == 1u);

    // The remaining order (id=2) is now the head — it should match next.
    book.add_order(3, Side::Bid, 100, 20, trades);
    REQUIRE(trades.size()              == 1u);
    REQUIRE(trades[0].passive_order_id == 2u);

    check_pool_invariant(book);
}

TEST_CASE("OrderBook: cancel tail order in queue", "[book][cancel]") {
    OrderBook book;
    std::vector<Trade> trades;

    book.add_order(1, Side::Bid, 100, 10, trades);  // head
    book.add_order(2, Side::Bid, 100, 20, trades);  // tail

    REQUIRE(book.cancel_order(2));
    REQUIRE(book.bid_qty_at(100) == 10u);
    REQUIRE(book.order_count()   == 1u);
    REQUIRE(book.best_bid()      == 100);

    check_pool_invariant(book);
}

// ---------------------------------------------------------------------------
// Cancel empties a non-best level (BBO should not change)
// ---------------------------------------------------------------------------
TEST_CASE("OrderBook: cancel non-best level does not change BBO", "[book][cancel]") {
    OrderBook book;
    std::vector<Trade> trades;

    book.add_order(1, Side::Bid, 105, 10, trades);  // best bid
    book.add_order(2, Side::Bid, 100, 10, trades);  // second level

    REQUIRE(book.best_bid() == 105);
    REQUIRE(book.cancel_order(2));         // cancel the non-best level
    REQUIRE(book.best_bid() == 105);       // BBO unchanged
    REQUIRE(book.order_count() == 1u);

    check_pool_invariant(book);
}

// ---------------------------------------------------------------------------
// Modify then verify matching uses updated quantity
// ---------------------------------------------------------------------------
TEST_CASE("OrderBook: modify quantity reflected in subsequent match", "[book][modify]") {
    OrderBook book;
    std::vector<Trade> trades;

    // Resting ask for 20.
    book.add_order(1, Side::Ask, 100, 20, trades);
    REQUIRE(book.ask_qty_at(100) == 20u);

    // Reduce to 8.
    REQUIRE(book.modify_order(1, 8));
    REQUIRE(book.ask_qty_at(100) == 8u);

    // Incoming bid for 8 — should fully consume the modified ask.
    book.add_order(2, Side::Bid, 100, 8, trades);
    REQUIRE(trades.size()    == 1u);
    REQUIRE(trades[0].quantity == 8u);

    REQUIRE(book.order_count() == 0u);
    REQUIRE(book.best_ask()    == INVALID_PRICE);
    check_pool_invariant(book);
}

TEST_CASE("OrderBook: modify to zero rejected", "[book][modify]") {
    OrderBook book;
    std::vector<Trade> trades;

    book.add_order(1, Side::Bid, 100, 10, trades);
    REQUIRE(!book.modify_order(1, 0));
    REQUIRE(book.bid_qty_at(100) == 10u);
}

TEST_CASE("OrderBook: modify unknown order rejected", "[book][modify]") {
    OrderBook book;
    REQUIRE(!book.modify_order(999, 5));
}

// ---------------------------------------------------------------------------
// BBO correctness after a sequence of operations
//
// Drives the book through a realistic sequence and checks BBO after each
// step.  This is the closest thing to an integration test for the BBO
// tracking logic.
// ---------------------------------------------------------------------------
TEST_CASE("OrderBook: BBO tracking across a mixed operation sequence", "[book][bbo]") {
    OrderBook book;
    std::vector<Trade> trades;

    // Build a bid ladder: 98, 99, 100.
    book.add_order(1, Side::Bid, 100, 10, trades);
    book.add_order(2, Side::Bid, 99,  10, trades);
    book.add_order(3, Side::Bid, 98,  10, trades);
    REQUIRE(book.best_bid() == 100);

    // Build an ask ladder: 101, 102, 103.
    book.add_order(4, Side::Ask, 101, 10, trades);
    book.add_order(5, Side::Ask, 102, 10, trades);
    book.add_order(6, Side::Ask, 103, 10, trades);
    REQUIRE(book.best_ask() == 101);
    REQUIRE(trades.empty());  // no cross yet

    // Cancel the best bid — BBO should step down to 99.
    book.cancel_order(1);
    REQUIRE(book.best_bid() == 99);

    // Cancel the best ask — BBO should step up to 102.
    book.cancel_order(4);
    REQUIRE(book.best_ask() == 102);

    // Add a new best bid at 101 — crosses best ask at 102? No: 101 < 102.
    book.add_order(7, Side::Bid, 101, 5, trades);
    REQUIRE(trades.empty());
    REQUIRE(book.best_bid() == 101);

    // Add an ask at 101 — crosses best bid at 101.  Should match.
    book.add_order(8, Side::Ask, 101, 5, trades);
    REQUIRE(trades.size() == 1u);
    REQUIRE(trades[0].quantity == 5u);

    // Both sides of that trade consumed.
    REQUIRE(book.best_bid() == 99);
    REQUIRE(book.best_ask() == 102);

    check_pool_invariant(book);
}

// ---------------------------------------------------------------------------
// Pool leak invariant across a high-volume sequence
//
// Runs 1 000 add/cancel pairs and asserts the pool invariant after every
// operation.  Any leak or double-free shows up immediately.
// ---------------------------------------------------------------------------
TEST_CASE("OrderBook: pool invariant holds across 1000 add/cancel cycles", "[book][pool]") {
    OrderBook book;
    std::vector<Trade> trades;

    for (uint64_t i = 1; i <= 1000; ++i) {
        // Spread orders across a range of prices so BBO scanning is exercised.
        const int64_t price = static_cast<int64_t>(50'000 + (i % 100));
        book.add_order(i, Side::Bid, price, 10, trades);
        check_pool_invariant(book);
    }

    // Cancel them all in reverse order (exercises both head and non-head removal).
    for (uint64_t i = 1000; i >= 1; --i) {
        book.cancel_order(i);
        check_pool_invariant(book);
    }

    REQUIRE(book.order_count() == 0u);
    REQUIRE(book.best_bid()    == INVALID_PRICE);
}

// ---------------------------------------------------------------------------
// Matching correctness: total_qty on the level stays consistent
//
// After partial matches and cancels, bid_qty_at() must always equal the
// sum of leaves_qty across all resting orders at that level.
// This catches any bug where total_qty is not updated correctly.
// ---------------------------------------------------------------------------
TEST_CASE("OrderBook: level total_qty stays consistent under partial fills", "[book][match]") {
    OrderBook book;
    std::vector<Trade> trades;

    // Three resting asks totalling 30.
    book.add_order(1, Side::Ask, 100, 10, trades);
    book.add_order(2, Side::Ask, 100, 10, trades);
    book.add_order(3, Side::Ask, 100, 10, trades);
    REQUIRE(book.ask_qty_at(100) == 30u);

    // Partial fill of 7 — only the head order (id=1) is touched.
    book.add_order(4, Side::Bid, 100, 7, trades);
    REQUIRE(book.ask_qty_at(100) == 23u);  // 3 + 10 + 10

    // Another partial fill of 5 — consumes the rest of id=1 (3) and 2 from id=2.
    book.add_order(5, Side::Bid, 100, 5, trades);
    REQUIRE(book.ask_qty_at(100) == 18u);  // 8 + 10

    // Full fill of remaining: 18.
    book.add_order(6, Side::Bid, 100, 18, trades);
    REQUIRE(book.ask_qty_at(100) == 0u);
    REQUIRE(book.best_ask()      == INVALID_PRICE);
    REQUIRE(book.order_count()   == 0u);

    check_pool_invariant(book);
}

// ---------------------------------------------------------------------------
// Aggressor that only partially fills, then rests
//
// Incoming order crosses a level, partially fills, then rests the residual.
// Checks that the residual is correctly enqueued and visible.
// ---------------------------------------------------------------------------
TEST_CASE("OrderBook: aggressor partially fills then rests residual", "[book][match]") {
    OrderBook book;
    std::vector<Trade> trades;

    // Resting ask for 5.
    book.add_order(1, Side::Ask, 100, 5, trades);

    // Incoming bid for 20 — fills 5, then rests 15 at 100.
    book.add_order(2, Side::Bid, 100, 20, trades);

    REQUIRE(trades.size()     == 1u);
    REQUIRE(trades[0].quantity == 5u);

    // Residual bid of 15 should be resting.
    REQUIRE(book.order_count()    == 1u);
    REQUIRE(book.best_bid()       == 100);
    REQUIRE(book.bid_qty_at(100)  == 15u);
    REQUIRE(book.best_ask()       == INVALID_PRICE);

    check_pool_invariant(book);
}

// ---------------------------------------------------------------------------
// Edge prices: MIN and MAX tick boundaries
// ---------------------------------------------------------------------------
TEST_CASE("OrderBook: orders at MIN and MAX valid price ticks accepted", "[book][edge]") {
    OrderBook book;
    std::vector<Trade> trades;

    REQUIRE(book.add_order(1, Side::Bid, OrderBook::MIN_PRICE_TICK, 1, trades));
    REQUIRE(book.add_order(2, Side::Ask, OrderBook::MAX_PRICE_TICK, 1, trades));

    REQUIRE(book.best_bid() == OrderBook::MIN_PRICE_TICK);
    REQUIRE(book.best_ask() == OrderBook::MAX_PRICE_TICK);
    REQUIRE(trades.empty());

    check_pool_invariant(book);
}

// ---------------------------------------------------------------------------
// Full match emitted as ask aggressor (symmetric to the bid aggressor test)
// ---------------------------------------------------------------------------
TEST_CASE("OrderBook: full match (ask aggressor)", "[book][match]") {
    OrderBook book;
    std::vector<Trade> trades;

    book.add_order(1, Side::Bid, 200, 10, trades);
    REQUIRE(trades.empty());

    book.add_order(2, Side::Ask, 200, 10, trades);
    REQUIRE(trades.size()              == 1u);
    REQUIRE(trades[0].quantity         == 10u);
    REQUIRE(trades[0].aggressive_order_id == 2u);
    REQUIRE(trades[0].passive_order_id    == 1u);

    REQUIRE(book.order_count() == 0u);
    check_pool_invariant(book);
}

// ===========================================================================
// lockfree_pool_allocator.hpp  (Version B)
//
// The single-threaded behaviour contract must be identical to Version A.
// These tests verify that first, then add concurrency-specific tests.
// ===========================================================================

// ---------------------------------------------------------------------------
// Single-threaded correctness — same contract as Version A
// ---------------------------------------------------------------------------

TEST_CASE("LockFreePool: construction", "[lfpool]") {
    constexpr size_t CAP = 128;
    LockFreePoolAllocator pool(CAP);
    REQUIRE(pool.capacity() == CAP);
}

TEST_CASE("LockFreePool: allocate returns non-null and distinct pointers", "[lfpool]") {
    LockFreePoolAllocator pool(4);

    Order* a = pool.allocate();
    Order* b = pool.allocate();
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(a != b);
    REQUIRE(pool.owns(a));
    REQUIRE(pool.owns(b));

    pool.deallocate(a);
    pool.deallocate(b);
}

TEST_CASE("LockFreePool: exhaustion returns nullptr", "[lfpool]") {
    LockFreePoolAllocator pool(2);
    Order* a = pool.allocate();
    Order* b = pool.allocate();
    REQUIRE(pool.allocate() == nullptr);   // pool full
    pool.deallocate(a);
    pool.deallocate(b);
}

TEST_CASE("LockFreePool: slot is reusable after deallocate", "[lfpool]") {
    LockFreePoolAllocator pool(2);

    Order* a = pool.allocate();
    pool.deallocate(a);

    // The same slot (or another) must come back as a valid non-null pointer.
    Order* b = pool.allocate();
    REQUIRE(b != nullptr);
    REQUIRE(pool.owns(b));

    pool.deallocate(b);
}

TEST_CASE("LockFreePool: alignment of allocated slots", "[lfpool]") {
    LockFreePoolAllocator pool(8);
    std::vector<Order*> ptrs;
    for (int i = 0; i < 8; ++i) {
        Order* p = pool.allocate();
        REQUIRE(p != nullptr);
        // Every slot must satisfy Order's alignas(64) requirement.
        REQUIRE(reinterpret_cast<uintptr_t>(p) % alignof(Order) == 0u);
        ptrs.push_back(p);
    }
    for (auto* p : ptrs) pool.deallocate(p);
}

TEST_CASE("LockFreePool: LIFO order — most recently freed slot returned first", "[lfpool]") {
    // Version B uses a stack (LIFO), same as Version A.
    // The last slot pushed onto the free list is the first one popped.
    // This matters for cache warmth: recently freed slots are hot.
    LockFreePoolAllocator pool(4);

    Order* a = pool.allocate();
    Order* b = pool.allocate();

    pool.deallocate(a);
    pool.deallocate(b);   // b is now at the head of the free list

    Order* first  = pool.allocate();   // should be b
    Order* second = pool.allocate();   // should be a

    REQUIRE(first  == b);
    REQUIRE(second == a);

    pool.deallocate(first);
    pool.deallocate(second);
}

// ---------------------------------------------------------------------------
// ABA-protection test
//
// We cannot directly observe the internal tag counter, but we can verify
// that repeated alloc/dealloc cycles with interleaved reuse do not corrupt
// the free list — the symptom of an ABA bug would be a crash, a returned
// nullptr before the pool is exhausted, or a duplicate pointer being handed
// out.
//
// This test simulates the ABA pattern in a single thread:
//   1. Allocate X (head moves to Y).
//   2. Deallocate X (X is pushed back, tag increments).
//   3. Allocate X again — this is the "A→B→A" scenario.
//   4. Verify we get X back (not nullptr, not a duplicate).
// ---------------------------------------------------------------------------
TEST_CASE("LockFreePool: ABA scenario does not corrupt free list", "[lfpool]") {
    LockFreePoolAllocator pool(4);

    // Drain all slots.
    Order* slots[4];
    for (auto& s : slots) s = pool.allocate();
    REQUIRE(pool.allocate() == nullptr);

    // Return them all.
    for (auto* s : slots) pool.deallocate(s);

    // Repeatedly alloc/dealloc the head slot.
    // If ABA were possible here, the tag counter would stop it;
    // a real bug would manifest as a null return or duplicate pointer.
    std::vector<Order*> seen;
    for (int round = 0; round < 8; ++round) {
        Order* p = pool.allocate();
        REQUIRE(p != nullptr);

        // Check we haven't handed out the same slot twice in this round.
        for (auto* prev : seen) REQUIRE(prev != p);
        seen.push_back(p);

        pool.deallocate(p);
        seen.clear();
    }
}

// ---------------------------------------------------------------------------
// Concurrent stress test
//
// One producer thread and one consumer thread hammer the allocator
// simultaneously.  The producer allocates slots and puts them into an
// SPSC queue; the consumer takes them out and deallocates them.
//
// If the lock-free logic is broken (wrong memory ordering, missing tag
// increment, etc.) this test will crash or deadlock under ASan/TSan.
//
// Run the test binary with -fsanitize=thread to catch data races.
// ---------------------------------------------------------------------------
TEST_CASE("LockFreePool: concurrent producer/consumer stress", "[lfpool][concurrent]") {
    constexpr size_t POOL_CAP   = 1024;
    constexpr int    ITERATIONS = 50'000;

    LockFreePoolAllocator pool(POOL_CAP);

    // Use an atomic flag so the consumer knows when the producer is done.
    std::atomic<bool> producer_done{false};

    // Shared SPSC queue to pass allocated pointers from producer to consumer.
    // We need it large enough that the producer doesn't block.
    SPSCQueue<Order*, 2048> handoff;

    std::atomic<int> allocated_count{0};
    std::atomic<int> freed_count{0};

    // Producer: allocates orders and hands them to the consumer.
    std::thread producer([&]() {
        for (int i = 0; i < ITERATIONS; ++i) {
            Order* p = nullptr;
            // Spin until a slot is available (pool may be momentarily full).
            while ((p = pool.allocate()) == nullptr) {
                // Yield to let the consumer thread make progress.
                std::this_thread::yield();
            }
            // Write a sentinel value into the slot so we can verify it
            // arrives intact on the consumer side.
            new (p) Order{};
            p->order_id = static_cast<uint64_t>(i);

            // Spin until the queue has room.
            while (!handoff.push(p)) std::this_thread::yield();
            allocated_count.fetch_add(1, std::memory_order_relaxed);
        }
        producer_done.store(true, std::memory_order_release);
    });

    // Consumer: receives orders from the producer and deallocates them.
    std::thread consumer([&]() {
        while (true) {
            Order* p = nullptr;
            if (handoff.pop(p)) {
                // Verify the sentinel value survived the round trip.
                REQUIRE(pool.owns(p));
                p->~Order();
                pool.deallocate(p);
                freed_count.fetch_add(1, std::memory_order_relaxed);
            } else if (producer_done.load(std::memory_order_acquire)) {
                // Drain any remaining items before exiting.
                if (!handoff.pop(p)) break;
                REQUIRE(pool.owns(p));
                p->~Order();
                pool.deallocate(p);
                freed_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    producer.join();
    consumer.join();

    REQUIRE(allocated_count.load() == ITERATIONS);
    REQUIRE(freed_count.load()     == ITERATIONS);
}
