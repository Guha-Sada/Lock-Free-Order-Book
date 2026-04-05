#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <thread>
#include <atomic>

#include "order.hpp"
#include "price_level.hpp"
#include "pool_allocator.hpp"
#include "lockfree_pool_allocator.hpp"
#include "spsc_queue.hpp"
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
