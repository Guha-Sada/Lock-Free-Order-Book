#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "order.hpp"
#include "price_level.hpp"
#include "pool_allocator.hpp"
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
