#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <cstdint>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <algorithm>   // nth_element used by LatencyRecorder (included here for clarity)

#include "order_book.hpp"
#include "flat_hash_map.hpp"
#include "pool_allocator.hpp"
#include "lockfree_pool_allocator.hpp"
#include "spsc_queue.hpp"
#include "naive_spsc_queue.hpp"

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
// SPSC Queue benchmark suite — three variants to tell the complete story
//
// 1. BM_SPSCQueue_Roundtrip        — single-threaded, padded queue
// 2. BM_SPSCQueue_Throughput       — two-thread,      padded queue
// 3. BM_NaiveSPSCQueue_Throughput  — two-thread,      naive (false-sharing)
//
// WHY THE SINGLE-THREADED BENCHMARK IS NOT ENOUGH:
//   In BM_SPSCQueue_Roundtrip, both push() and pop() run on the same core.
//   There is no second core to fight over the cache line, so false sharing
//   cannot occur — write_pos_ and read_pos_ never bounce between cores.
//   The padded and naive queues will show nearly identical numbers here.
//
//   The two-thread benchmarks (2 and 3) run producer and consumer on
//   separate cores simultaneously.  Now false sharing is real: every push
//   by the producer invalidates the consumer's cached copy of the cache
//   line holding read_pos_, and vice versa.  The naive queue pays this
//   penalty on every single operation.
//
// EXPECTED RESULTS (modern desktop CPU, 2 physical cores):
//   BM_SPSCQueue_Roundtrip        ~5–10 ns/op   (same-core, no contention)
//   BM_SPSCQueue_Throughput       ~8–15 ns/op   (cross-core, no false share)
//   BM_NaiveSPSCQueue_Throughput  ~30–80 ns/op  (cross-core, false sharing)
//
//   The ratio between (2) and (3) is your empirical false-sharing penalty.
//   Put both numbers in the README.
// ---------------------------------------------------------------------------

// 1. Single-threaded roundtrip — baseline, shows raw atomic op overhead
static void BM_SPSCQueue_Roundtrip(benchmark::State& state) {
    SPSCQueue<uint64_t, 4096> q;
    uint64_t val = 0;

    for (auto _ : state) {
        while (!q.push(val)) {}
        while (!q.pop(val))  {}
        ++val;
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("padded, single-threaded");
}
BENCHMARK(BM_SPSCQueue_Roundtrip)->Unit(benchmark::kNanosecond);

// ---------------------------------------------------------------------------
// Two-thread throughput helper — used by both the padded and naive variants.
//
// The benchmark loop IS the producer.  A background consumer thread drains
// the queue as fast as possible.  We measure how many items per second the
// producer can push when a real consumer is competing on another core.
//
// The stop flag uses relaxed ordering for the poll loop (perf-critical) and
// acquire/release for the final handshake (correctness-critical).
// ---------------------------------------------------------------------------
template<typename Queue>
static void run_throughput_bench(benchmark::State& state, Queue& q) {
    std::atomic<bool> stop{false};

    std::thread consumer([&]() {
        uint64_t val{};
        while (!stop.load(std::memory_order_relaxed))
            q.pop(val);
        // Drain any remaining items after the producer signals stop.
        while (q.pop(val)) {}
    });

    uint64_t val = 0;
    for (auto _ : state) {
        // Spin until the queue has room.  In a real system you would
        // use a different backpressure strategy (drop, batch, etc.).
        while (!q.push(val)) std::this_thread::yield();
        benchmark::DoNotOptimize(val);
        ++val;
    }

    stop.store(true, std::memory_order_release);
    consumer.join();

    state.SetItemsProcessed(state.iterations());
}

// 2. Padded queue — write_pos_ and read_pos_ on separate cache lines
static void BM_SPSCQueue_Throughput(benchmark::State& state) {
    SPSCQueue<uint64_t, 4096> q;
    run_throughput_bench(state, q);
    state.SetLabel("padded (no false sharing)");
}
BENCHMARK(BM_SPSCQueue_Throughput)->Unit(benchmark::kNanosecond);

// 3. Naive queue — write_pos_ and read_pos_ on the SAME cache line
//
// This is the control: identical logic, identical memory ordering,
// only difference is the missing alignas(64).  Any throughput gap
// between this and BM_SPSCQueue_Throughput is purely false sharing.
static void BM_NaiveSPSCQueue_Throughput(benchmark::State& state) {
    NaiveSPSCQueue<uint64_t, 4096> q;
    run_throughput_bench(state, q);
    state.SetLabel("naive (false sharing)");
}
BENCHMARK(BM_NaiveSPSCQueue_Throughput)->Unit(benchmark::kNanosecond);

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

// ---------------------------------------------------------------------------
// Phase 6 — Percentile Latency Suite
//
// The benchmarks above measure THROUGHPUT: how many operations per second the
// system can sustain.  Quant firms care equally (or more) about TAIL LATENCY:
// what happens at p99 and p999.
//
// The distinction matters because:
//   - A throughput benchmark averages over millions of iterations.  A 10 µs
//     spike that happens once every 1000 iterations is invisible in the
//     average but is very visible to a trading strategy that needs sub-100 µs
//     determinism.
//   - p99 = the latency exceeded only 1% of the time.
//   - p999 = the latency exceeded only 0.1% of the time.
//   - "My add_order is 50 ns p99" is a much stronger statement than
//     "my add_order averages 5 ns/op" when a competitor might have the same
//     average but a 5 µs p999 due to occasional cache misses or lock waits.
//
// IMPLEMENTATION STRATEGY:
//
//   1. now_ns() — a thin wrapper around clock_gettime(CLOCK_MONOTONIC_RAW).
//      MONOTONIC_RAW is not adjusted by NTP, so it never jumps backward or
//      forward mid-benchmark.  On Linux this resolves to a VDSO call (no
//      syscall overhead); on macOS it resolves to clock_gettime_nsec_np.
//
//   2. LatencyRecorder — records individual sample times into a pre-allocated
//      std::vector<int64_t>.  The vector is reserved BEFORE the benchmark loop
//      runs so that push_back() never allocates during measurement.  After the
//      loop, std::nth_element (O(n) average) is used to compute p50/p99/p999
//      without a full O(n log n) sort.
//
//   3. Three benchmark functions, each measuring a distinct operation:
//        BM_Latency_AddOrder  — add_order() with no match (resting ask, bid side)
//        BM_Latency_Cancel    — cancel_order() on a pre-inserted order
//        BM_Latency_Match     — add_order() that immediately crosses a resting ask
//
//   4. Results are reported via state.counters["p50_ns"], ["p99_ns"], ["p999_ns"].
//      google/benchmark prints these as extra columns in the results table.
//
// NOTE ON MEASUREMENT ACCURACY:
//   clock_gettime itself costs ~10–30 ns on modern hardware.  That overhead
//   is included in every sample here — it is part of the "real wall-clock
//   cost of one operation including measurement overhead."  For a pure
//   function-call latency you would use RDTSC instead, but RDTSC requires
//   careful frequency calibration.  For interview/portfolio purposes,
//   clock_gettime is the right tradeoff: accurate enough, portable, honest.
// ---------------------------------------------------------------------------

// Returns current time in nanoseconds from a monotonic clock.
// CLOCK_MONOTONIC_RAW: not adjusted by NTP — no backward jumps.
// On Linux x86/ARM servers this is a VDSO call with ~1 ns resolution.
//
// APPLE SILICON LIMITATION:
//   On macOS ARM64, clock_gettime() reads cntvct_el0 — the ARM virtual
//   counter — which ticks at 24 MHz (one tick every ~41.7 ns).
//   Any operation faster than ~42 ns measures as 0 ns.
//   The pmccntr_el0 CPU cycle counter has nanosecond resolution but
//   requires root (EL0 access disabled by macOS by default).
//
//   FIX: batch timing via BENCH_BATCH below.  Time N operations per
//   sample and divide by N.  With N=16 the floor drops to ~2.6 ns,
//   well below any real operation we care to measure.
static inline int64_t now_ns() noexcept {
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;
}

// Operations batched per timing sample.
// Chosen so that (42 ns timer tick / BENCH_BATCH) ≈ 2.6 ns floor,
// below the cost of any operation we benchmark.
static constexpr int BENCH_BATCH = 16;

// ---------------------------------------------------------------------------
// LatencyRecorder
//
// Records individual latency samples during a benchmark run, then computes
// percentiles at the end.  All allocation happens in the constructor; the
// record() call is a single push_back — branch-predicted-taken, ~1 ns.
//
// Percentile computation:
//   std::nth_element rearranges the vector so that element at index k is the
//   value that would be there in a sorted array, in O(n) average time.
//   We call it three times (p50, p99, p999), each on a fresh copy-view
//   of the data.  This is faster than std::sort (O(n log n)) when n is large.
//
// Why not use google/benchmark's built-in statistics?
//   benchmark::StatisticsFunc computes mean/stddev, not percentiles.  We
//   need true order-statistic percentiles, so we roll our own.
// ---------------------------------------------------------------------------
struct LatencyRecorder {
    std::vector<int64_t> samples;

    explicit LatencyRecorder(size_t capacity) {
        samples.reserve(capacity);
    }

    // Called inside the hot loop — must be cheap.
    inline void record(int64_t latency_ns) noexcept {
        samples.push_back(latency_ns);
    }

    // Compute the value at the given percentile (0.0 – 1.0).
    // Uses nth_element: O(n) average, mutates the vector.
    int64_t percentile(double p) {
        if (samples.empty()) return 0;
        const size_t idx = static_cast<size_t>(p * static_cast<double>(samples.size() - 1));
        std::nth_element(samples.begin(),
                         samples.begin() + static_cast<ptrdiff_t>(idx),
                         samples.end());
        return samples[idx];
    }

    // Publish p50 / p99 / p999 into benchmark state counters.
    // Call this AFTER the benchmark loop ends.
    void publish(benchmark::State& state) {
        // nth_element mutates the vector, so compute in order from largest
        // percentile to smallest to avoid partial-sort invalidation.
        // Actually nth_element only guarantees elements below idx are <=
        // the pivot and elements above are >=; each call is independent,
        // so order doesn't matter for correctness.  Largest first is cleaner.
        int64_t p999 = percentile(0.999);
        int64_t p99  = percentile(0.99);
        int64_t p50  = percentile(0.50);

        // benchmark::Counter::kAvgThreads is the default; use plain value.
        state.counters["p50_ns"]  = static_cast<double>(p50);
        state.counters["p99_ns"]  = static_cast<double>(p99);
        state.counters["p999_ns"] = static_cast<double>(p999);
    }
};

// ---------------------------------------------------------------------------
// BM_Latency_AddOrder
//
// Measures the wall-clock latency of a single add_order() call that does NOT
// trigger any matching.  The ask is placed 200 ticks above the best bid so
// the match engine exits immediately after the price check.
//
// What this isolates:
//   - pool_.allocate()                  (free-list pop)
//   - placement-new of Order            (memset-like init)
//   - order_map_.emplace()              (hash table insert)
//   - match() early exit (no liquidity crossing)
//   - enqueue() onto the ask level      (doubly-linked list append)
//   - best_ask_ update                  (conditional branch + store)
//
// The subsequent cancel cleans up so the pool never exhausts and the
// order_map_ doesn't grow without bound.
// ---------------------------------------------------------------------------
static void BM_Latency_AddOrder(benchmark::State& state) {
    OrderBook book;
    std::vector<Trade> trades;
    trades.reserve(BENCH_BATCH);

    const int64_t MID = 50'000;
    for (int64_t p = MID - 100; p <= MID; ++p) {
        book.add_order(static_cast<uint64_t>(p), Side::Bid, p, 100, trades);
    }

    LatencyRecorder rec(4'000'000 / BENCH_BATCH + 1);
    uint64_t order_id = 2'000'000;

    for (auto _ : state) {
        trades.clear();

        // Time BENCH_BATCH add_orders in one shot, then divide.
        // This beats the ~42 ns Apple Silicon timer floor.
        const int64_t t0 = now_ns();
        for (int i = 0; i < BENCH_BATCH; ++i) {
            book.add_order(order_id + i, Side::Ask,
                           MID + 200 + static_cast<int64_t>((order_id + i) % 50),
                           10, trades);
        }
        const int64_t t1 = now_ns();

        rec.record((t1 - t0) / BENCH_BATCH);

        // Cancel all — outside the timed region.
        for (int i = 0; i < BENCH_BATCH; ++i) {
            book.cancel_order(order_id + i);
        }
        order_id += BENCH_BATCH;
    }

    rec.publish(state);
    state.SetItemsProcessed(state.iterations());
    state.SetLabel("add_order (no match, batched x16)");
}
BENCHMARK(BM_Latency_AddOrder)->Unit(benchmark::kNanosecond);

// ---------------------------------------------------------------------------
// BM_Latency_Cancel
//
// Measures the wall-clock latency of cancel_order() in isolation.
//
// What this isolates:
//   - order_map_.find()                 (hash table lookup)
//   - order_map_.erase()                (hash table remove)
//   - dequeue()                         (doubly-linked list splice)
//   - update_best_bid_after_remove()    (conditional level scan)
//   - pool_.deallocate()                (free-list push)
//
// To keep the hot path realistic, we pre-insert a fresh batch of orders
// before starting the loop and refill whenever the batch runs out.  The
// refill is excluded from measurement.
// ---------------------------------------------------------------------------
static void BM_Latency_Cancel(benchmark::State& state) {
    constexpr int FILL_SIZE = 1024;
    OrderBook book;
    std::vector<Trade> trades;
    trades.reserve(1);

    auto prices = make_prices(FILL_SIZE, 50'000, 200);
    uint64_t next_id = 1;
    std::vector<uint64_t> live_ids;
    live_ids.reserve(FILL_SIZE);

    auto refill = [&]() {
        for (int i = 0; i < FILL_SIZE; ++i) {
            trades.clear();
            book.add_order(next_id, Side::Bid,
                           prices[static_cast<size_t>(i) % FILL_SIZE],
                           10, trades);
            live_ids.push_back(next_id++);
        }
    };

    refill();
    size_t idx = 0;
    LatencyRecorder rec(4'000'000 / BENCH_BATCH + 1);

    for (auto _ : state) {
        // Refill in multiples of BENCH_BATCH so we never straddle a refill
        // boundary mid-batch.
        if (idx + BENCH_BATCH > live_ids.size()) {
            live_ids.clear();
            refill();
            idx = 0;
        }

        const int64_t t0 = now_ns();
        for (int i = 0; i < BENCH_BATCH; ++i) {
            book.cancel_order(live_ids[idx + i]);
        }
        const int64_t t1 = now_ns();

        rec.record((t1 - t0) / BENCH_BATCH);
        idx += BENCH_BATCH;
    }

    rec.publish(state);
    state.SetItemsProcessed(state.iterations());
    state.SetLabel("cancel_order (batched x16)");
}
BENCHMARK(BM_Latency_Cancel)->Unit(benchmark::kNanosecond);

// ---------------------------------------------------------------------------
// BM_Latency_Match
//
// Measures the wall-clock latency of a FULL crossing match:
//   1. add_order(ask at MID)   — rests on the book
//   2. add_order(bid at MID)   — crosses the ask, triggers match()
//
// We time only step 2 (the aggressor add_order), which is the hot path:
//   - alloc + placement-new
//   - order_map_.emplace()
//   - match() loop: one iteration, fill(), dequeue(), erase(), deallocate()
//   - update_best_ask_after_remove()
//   - aggressor fully filled: erase() + deallocate() of aggressor
//
// This is the most realistic single-operation latency for a market-order
// execution in a live system.
// ---------------------------------------------------------------------------
static void BM_Latency_Match(benchmark::State& state) {
    OrderBook book;
    std::vector<Trade> trades;
    trades.reserve(BENCH_BATCH);

    const int64_t MID = 50'000;
    uint64_t id = 1;
    LatencyRecorder rec(4'000'000 / BENCH_BATCH + 1);

    for (auto _ : state) {
        // Place BENCH_BATCH resting asks — NOT timed.
        trades.clear();
        for (int i = 0; i < BENCH_BATCH; ++i) {
            book.add_order(id++, Side::Ask, MID, 10, trades);
        }

        // Time BENCH_BATCH crossing bids.
        trades.clear();
        const int64_t t0 = now_ns();
        for (int i = 0; i < BENCH_BATCH; ++i) {
            book.add_order(id++, Side::Bid, MID, 10, trades);
        }
        const int64_t t1 = now_ns();

        rec.record((t1 - t0) / BENCH_BATCH);
        benchmark::DoNotOptimize(trades.data());
    }

    rec.publish(state);
    state.SetItemsProcessed(state.iterations());
    state.SetLabel("full match (bid crosses ask, batched x16)");
}
BENCHMARK(BM_Latency_Match)->Unit(benchmark::kNanosecond);

// ---------------------------------------------------------------------------
// Phase 7 — HashMap Comparison: std::unordered_map vs FlatHashMap
//
// These benchmarks isolate exactly the access pattern that OrderBook uses:
//   1. insert(id, ptr)   — on every add_order
//   2. find(id)          — on cancel, modify, and duplicate-check
//   3. erase(id)         — on cancel and fully-matched add
//
// They are NOT full OrderBook benchmarks — they only measure the cost of the
// hash map operations themselves.  This makes the comparison clean: there is
// no pool allocation, no linked-list manipulation, no price level scanning.
// The ONLY variable is the hash map implementation.
//
// WHY std::unordered_map IS SLOWER AT p99/p999:
//   std::unordered_map<K,V> is a node-based container.  Each insert()
//   allocates a new heap node (malloc).  Each find() follows a pointer from
//   the bucket array to the first node in the chain, then potentially follows
//   more next-pointers if there are collisions.  Each of these pointer follows
//   is a potential cache miss because the nodes are scattered across the heap.
//
//   Under steady-state order-book traffic (insert + erase cycling through the
//   same address range), malloc's free-list reuses recently freed nodes, so
//   the AVERAGE case looks fast.  But occasionally a node lands in a cold
//   cache line and you pay the full ~100 ns LLC miss.  That is your p99 spike.
//
// WHY FlatHashMap IS FASTER:
//   All slots are in a single contiguous array.  Linear probing means the CPU
//   fetches neighbouring slots as part of the same cache line.  A 64-byte
//   cache line holds 4 Slots (each 16 bytes), so a probe of 1-4 slots costs
//   at most one cache line load.  No heap allocation on insert.  No pointer
//   chasing.  p99 and p999 converge toward p50 because the variance source
//   (heap pointer scatter) is gone.
//
// HOW TO READ THE NUMBERS:
//   BM_HashMap_Unordered_*   — baseline (std::unordered_map)
//   BM_HashMap_Flat_*        — optimized (FlatHashMap)
//
//   Compare p50_ns, p99_ns, p999_ns columns.
//   A 2–5× improvement in p99/p999 with similar p50 is typical.
// ---------------------------------------------------------------------------

// ---- std::unordered_map baseline ----

// ---------------------------------------------------------------------------
// HashMap benchmark parameters
//
// MAP_BENCH_BATCH = 128:
//   Larger than the general BENCH_BATCH=16 because isolated hash map ops
//   are faster than full OrderBook ops.  With flat_find taking ~2 ns/op,
//   we need 128 * 2 ns = 256 ns per batch to clear the 42 ns Apple Silicon
//   timer floor with headroom (256 / 42 ≈ 6 ticks → floor ≈ 2 ns/op).
//
// MAP_LIVE = 8192:
//   Enough entries that the unordered_map's heap nodes (8192 * ~48 bytes
//   ≈ 393 KB) no longer fit in L1 data cache (typically 64–128 KB per
//   core).  This creates genuine cache pressure: the hash bucket array is
//   in L1 but the node pointer from the bucket leads to an L2/L3 access.
//   FlatHashMap's array (8192 * 16 bytes = 128 KB) is larger but still
//   contiguous, so a prefetcher can load ahead — sequential or linear-probe
//   access costs one cache line per 4 slots, not one per slot.
//
// RANDOM KEY ACCESS:
//   Sequential `(key + i) % MAP_LIVE` access is too predictable — both
//   maps benefit equally from hardware prefetching.  A pre-shuffled key
//   permutation defeats prefetching for unordered_map's scattered heap
//   nodes (the prefetcher cannot predict a heap address from the key) while
//   FlatHashMap's probing still stays within its contiguous slot array.
//   This is the pattern a real order book sees: cancel requests arrive for
//   arbitrary live orders, not in insertion order.
// ---------------------------------------------------------------------------
static constexpr int      MAP_BENCH_BATCH = 128;
static constexpr uint64_t MAP_LIVE        = 8192;

// Build a random permutation of [0, MAP_LIVE) once.  Used by both find
// benchmarks so the access pattern is identical across implementations.
static std::vector<uint64_t> make_random_keys() {
    std::vector<uint64_t> keys(MAP_LIVE);
    for (uint64_t i = 0; i < MAP_LIVE; ++i) keys[i] = i;
    std::mt19937_64 rng(12345);
    std::shuffle(keys.begin(), keys.end(), rng);
    return keys;
}

static void BM_HashMap_Unordered_Insert(benchmark::State& state) {
    std::unordered_map<uint64_t, int> m;
    m.reserve(MAP_LIVE * 2);
    int val = 42;

    for (uint64_t k = 0; k < MAP_LIVE; ++k) m.emplace(k, val);

    LatencyRecorder rec(4'000'000 / MAP_BENCH_BATCH + 1);
    uint64_t insert_key = MAP_LIVE;
    uint64_t erase_key  = 0;

    for (auto _ : state) {
        const int64_t t0 = now_ns();
        for (int i = 0; i < MAP_BENCH_BATCH; ++i) {
            m.emplace(insert_key + i, val);
        }
        const int64_t t1 = now_ns();
        benchmark::DoNotOptimize(m.size());

        for (int i = 0; i < MAP_BENCH_BATCH; ++i) m.erase(erase_key + i);

        rec.record((t1 - t0) / MAP_BENCH_BATCH);
        insert_key += MAP_BENCH_BATCH;
        erase_key  += MAP_BENCH_BATCH;
    }
    rec.publish(state);
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HashMap_Unordered_Insert)->Unit(benchmark::kNanosecond);

static void BM_HashMap_Unordered_Find(benchmark::State& state) {
    std::unordered_map<uint64_t, int> m;
    m.reserve(MAP_LIVE * 2);
    for (uint64_t k = 0; k < MAP_LIVE; ++k) m.emplace(k, static_cast<int>(k));

    // Random key permutation — defeats hardware prefetching for heap nodes.
    const auto keys = make_random_keys();
    LatencyRecorder rec(4'000'000 / MAP_BENCH_BATCH + 1);
    size_t idx = 0;

    for (auto _ : state) {
        const int64_t t0 = now_ns();
        for (int i = 0; i < MAP_BENCH_BATCH; ++i) {
            auto it = m.find(keys[(idx + i) % MAP_LIVE]);
            benchmark::DoNotOptimize(it);
        }
        const int64_t t1 = now_ns();

        rec.record((t1 - t0) / MAP_BENCH_BATCH);
        idx += MAP_BENCH_BATCH;
    }
    rec.publish(state);
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HashMap_Unordered_Find)->Unit(benchmark::kNanosecond);

// ---- FlatHashMap optimized ----

static void BM_HashMap_Flat_Insert(benchmark::State& state) {
    // FlatHashMap capacity: 4× MAP_LIVE to keep load factor < 25%.
    FlatHashMap<int> m(MAP_LIVE * 4);
    int val = 42;

    for (uint64_t k = 0; k < MAP_LIVE; ++k) m.insert(k, &val);

    LatencyRecorder rec(4'000'000 / MAP_BENCH_BATCH + 1);
    uint64_t insert_key = MAP_LIVE;
    uint64_t erase_key  = 0;

    for (auto _ : state) {
        const int64_t t0 = now_ns();
        for (int i = 0; i < MAP_BENCH_BATCH; ++i) {
            m.insert(insert_key + i, &val);
        }
        const int64_t t1 = now_ns();
        benchmark::DoNotOptimize(m.size());

        for (int i = 0; i < MAP_BENCH_BATCH; ++i) m.erase(erase_key + i);

        rec.record((t1 - t0) / MAP_BENCH_BATCH);
        insert_key += MAP_BENCH_BATCH;
        erase_key  += MAP_BENCH_BATCH;
    }
    rec.publish(state);
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HashMap_Flat_Insert)->Unit(benchmark::kNanosecond);

static void BM_HashMap_Flat_Find(benchmark::State& state) {
    FlatHashMap<int> m(MAP_LIVE * 4);
    std::vector<int> arr(MAP_LIVE);
    for (uint64_t k = 0; k < MAP_LIVE; ++k) {
        arr[k] = static_cast<int>(k);
        m.insert(k, &arr[k]);
    }

    // Same random permutation as the unordered_map find benchmark so the
    // only variable is the data structure, not the access pattern.
    const auto keys = make_random_keys();
    LatencyRecorder rec(4'000'000 / MAP_BENCH_BATCH + 1);
    size_t idx = 0;

    for (auto _ : state) {
        const int64_t t0 = now_ns();
        for (int i = 0; i < MAP_BENCH_BATCH; ++i) {
            int** p = m.find(keys[(idx + i) % MAP_LIVE]);
            benchmark::DoNotOptimize(p);
        }
        const int64_t t1 = now_ns();

        rec.record((t1 - t0) / MAP_BENCH_BATCH);
        idx += MAP_BENCH_BATCH;
    }
    rec.publish(state);
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HashMap_Flat_Find)->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
