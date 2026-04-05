#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <cstdint>

#include "order_book.hpp"
#include "pool_allocator.hpp"
#include "spsc_queue.hpp"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Generates a repeatable sequence of prices around a mid-point.
// Using a fixed seed keeps benchmark runs comparable.
static std::vector<int64_t> make_prices(size_t n, int64_t mid = 50'000,
                                         int64_t spread = 500)
{
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int64_t> dist(mid - spread, mid + spread);
    std::vector<int64_t> prices(n);
    for (auto& p : prices) p = dist(rng);
    return prices;
}

// ---------------------------------------------------------------------------
// BM_AddOrder: latency of adding a single resting order
//
// The book is pre-populated with N orders spread across the bid side so that
// the incoming orders (asks above the market) never match.  This isolates the
// add_order() code path from the match engine.
// ---------------------------------------------------------------------------
static void BM_AddOrder(benchmark::State& state) {
    OrderBook book;
    std::vector<Trade> trades;
    trades.reserve(64);

    // Pre-populate bids so best_bid_ is set and the book is non-trivial.
    const int64_t mid = 50'000;
    for (int64_t p = mid - 100; p <= mid; ++p) {
        book.add_order(static_cast<uint64_t>(p), Side::Bid, p, 100, trades);
    }

    uint64_t order_id = 1'000'000;
    int64_t  price    = mid + 200;  // above market — never matches

    for (auto _ : state) {
        trades.clear();

        // Add then immediately cancel so the pool never exhausts.
        book.add_order(order_id, Side::Ask, price, 10, trades);
        book.cancel_order(order_id);
        ++order_id;

        // Vary price slightly to stress different price levels.
        price = mid + 200 + static_cast<int64_t>(order_id % 50);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_AddOrder)->Unit(benchmark::kNanosecond);

// ---------------------------------------------------------------------------
// BM_CancelOrder: latency of cancelling a resting order
// ---------------------------------------------------------------------------
static void BM_CancelOrder(benchmark::State& state) {
    constexpr int  BATCH = 1024;
    OrderBook book;
    std::vector<Trade> trades;
    trades.reserve(1);

    auto prices = make_prices(BATCH, 50'000, 200);

    uint64_t next_id = 1;

    // Pre-fill a batch of orders to cancel during the benchmark.
    std::vector<uint64_t> live_ids;
    live_ids.reserve(BATCH);

    auto refill = [&]() {
        for (int i = 0; i < BATCH; ++i) {
            trades.clear();
            book.add_order(next_id, Side::Bid, prices[static_cast<size_t>(i)],
                           10, trades);
            live_ids.push_back(next_id++);
        }
    };

    refill();
    size_t idx = 0;

    for (auto _ : state) {
        if (idx >= live_ids.size()) {
            live_ids.clear();
            refill();
            idx = 0;
        }
        book.cancel_order(live_ids[idx++]);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CancelOrder)->Unit(benchmark::kNanosecond);

// ---------------------------------------------------------------------------
// BM_PoolAllocatorVsMalloc: demonstrates why we have a pool allocator.
//
// Phase 3 task: after building PoolAllocator Version B (lock-free), add a
// third benchmark variant here and plot all three in the README.
// ---------------------------------------------------------------------------
static void BM_PoolAllocator(benchmark::State& state) {
    PoolAllocator pool(1u << 14);  // 16 384 slots

    for (auto _ : state) {
        Order* o = pool.allocate();
        benchmark::DoNotOptimize(o);  // prevent dead-code elimination
        pool.deallocate(o);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("pool");
}
BENCHMARK(BM_PoolAllocator)->Unit(benchmark::kNanosecond);

static void BM_SystemMalloc(benchmark::State& state) {
    for (auto _ : state) {
        // aligned_alloc mimics what the pool does, so the comparison is fair.
        void* p = std::aligned_alloc(alignof(Order), sizeof(Order));
        benchmark::DoNotOptimize(p);
        std::free(p);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("malloc");
}
BENCHMARK(BM_SystemMalloc)->Unit(benchmark::kNanosecond);

// ---------------------------------------------------------------------------
// BM_SPSCQueue_WithPadding / WithoutPadding
//
// Demonstrates the false-sharing penalty.  Phase 5 task: implement a
// "NaiveSPSCQueue" that places head_ and tail_ on the same cache line, then
// re-run this benchmark and add both results to the README.
// ---------------------------------------------------------------------------
static void BM_SPSCQueue_Roundtrip(benchmark::State& state) {
    // Single-threaded round-trip (push + pop) to measure the raw overhead
    // of the atomic operations and memory ordering.
    SPSCQueue<uint64_t, 4096> q;
    uint64_t val = 0;

    for (auto _ : state) {
        q.push(val);
        q.pop(val);
        ++val;
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SPSCQueue_Roundtrip)->Unit(benchmark::kNanosecond);

// ---------------------------------------------------------------------------
// BM_FullMatch: throughput of the match engine under continuous crossing flow
// ---------------------------------------------------------------------------
static void BM_FullMatch(benchmark::State& state) {
    OrderBook book;
    std::vector<Trade> trades;
    trades.reserve(4);

    const int64_t MID = 50'000;
    uint64_t id = 1;

    for (auto _ : state) {
        trades.clear();
        // Resting ask at MID
        book.add_order(id++, Side::Ask, MID, 10, trades);
        trades.clear();
        // Crossing bid — triggers a full match and removes both orders.
        book.add_order(id++, Side::Bid, MID, 10, trades);
        benchmark::DoNotOptimize(trades.data());
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_FullMatch)->Unit(benchmark::kNanosecond);
