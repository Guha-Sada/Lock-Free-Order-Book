#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <cstdint>
#include <thread>
#include <atomic>

#include "order_book.hpp"
#include "pool_allocator.hpp"
#include "lockfree_pool_allocator.hpp"
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
// Allocator comparison suite
//
// Three benchmarks run an identical workload — one alloc + one dealloc per
// iteration — using each allocator variant.  Run them together and compare
// the ns/op column in the output.
//
// Expected results on a modern x86-64 Linux machine:
//   BM_SystemMalloc          ~60–100 ns  (glibc ptmalloc2, lock + bookkeeping)
//   BM_PoolAllocator_A       ~3–5   ns  (plain pointer swap, no atomics)
//   BM_PoolAllocatorLF_B     ~8–15  ns  (CMPXCHG16B, single thread, no contention)
//
// Key insight: lock-free (Version B) is faster than system malloc but
// SLOWER than single-threaded Version A.  "Lock-free" does not mean
// "fastest possible" — it means "no thread can be indefinitely blocked."
// You pay for the atomic instruction even when there is no contention.
// ---------------------------------------------------------------------------

// Version A — single-threaded slab allocator (plain pointer swap)
static void BM_PoolAllocator_A(benchmark::State& state) {
    PoolAllocator pool(1u << 14);

    for (auto _ : state) {
        Order* o = pool.allocate();
        benchmark::DoNotOptimize(o);
        pool.deallocate(o);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("Version A (single-threaded)");
}
BENCHMARK(BM_PoolAllocator_A)->Unit(benchmark::kNanosecond);

// Version B — lock-free slab allocator (CMPXCHG16B tagged pointer)
static void BM_PoolAllocatorLF_B(benchmark::State& state) {
    LockFreePoolAllocator pool(1u << 14);

    for (auto _ : state) {
        Order* o = pool.allocate();
        benchmark::DoNotOptimize(o);
        pool.deallocate(o);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("Version B (lock-free)");
}
BENCHMARK(BM_PoolAllocatorLF_B)->Unit(benchmark::kNanosecond);

// System allocator — aligned_alloc/free (baseline to beat)
static void BM_SystemMalloc(benchmark::State& state) {
    for (auto _ : state) {
        void* p = std::aligned_alloc(alignof(Order), sizeof(Order));
        benchmark::DoNotOptimize(p);
        std::free(p);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("system malloc");
}
BENCHMARK(BM_SystemMalloc)->Unit(benchmark::kNanosecond);

// ---------------------------------------------------------------------------
// BM_PoolAllocatorLF_Contended
//
// Measures Version B under real contention: a background thread continuously
// deallocates while the benchmark thread allocates.  This is the scenario
// Version B is actually designed for.
//
// Under contention you will see CAS retries increase, pushing latency up
// relative to the uncontended BM_PoolAllocatorLF_B above.  The gap between
// the two is your "contention tax" — a number worth putting in the README.
// ---------------------------------------------------------------------------
static void BM_PoolAllocatorLF_Contended(benchmark::State& state) {
    constexpr size_t CAP = 1u << 14;
    LockFreePoolAllocator pool(CAP);

    // Pre-fill a side stash that the background thread will draw from.
    // We keep it separate from the benchmark thread's allocations.
    constexpr size_t STASH = CAP / 2;
    std::vector<Order*> stash;
    stash.reserve(STASH);
    for (size_t i = 0; i < STASH; ++i)
        stash.push_back(pool.allocate());

    std::atomic<bool> stop{false};

    // Background thread: continuously deallocates and reallocates from stash,
    // creating contention on the atomic free_head_.
    std::thread background([&]() {
        size_t idx = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            pool.deallocate(stash[idx % STASH]);
            stash[idx % STASH] = pool.allocate();
            if (!stash[idx % STASH]) {
                // Pool temporarily exhausted — just skip this slot.
                stash[idx % STASH] = pool.allocate();
            }
            ++idx;
        }
    });

    for (auto _ : state) {
        Order* o = pool.allocate();
        benchmark::DoNotOptimize(o);
        if (o) pool.deallocate(o);
    }

    stop.store(true, std::memory_order_release);
    background.join();

    // Return stash to pool (cleanup).
    for (auto* p : stash) if (p) pool.deallocate(p);

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("Version B (contended)");
}
BENCHMARK(BM_PoolAllocatorLF_Contended)->Unit(benchmark::kNanosecond);

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
