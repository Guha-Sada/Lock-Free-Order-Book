# Low-Latency Order Book

A price-time-priority central limit order book (CLOB) in C++20, built to demonstrate low-latency systems design for quant and robotics roles. The project covers the full stack: data layout, allocation, matching, lock-free concurrency, and a benchmark suite that measures p50/p99/p999 nanosecond latencies.

---

## Architecture

```
Producer thread                     Book thread
──────────────                     ────────────
OrderEvent  ──►  SPSCQueue  ──►   OrderBook
                (lock-free,        ┌─────────────────────────────┐
                 ring buffer)      │  PoolAllocator              │
                                   │    └─ aligned_alloc block    │
                                   │       free-list in free slots│
                                   │                             │
                                   │  bids_[NUM_LEVELS]          │
                                   │  asks_[NUM_LEVELS]          │
                                   │    └─ PriceLevel[price]     │
                                   │       head/tail queue       │
                                   │       total_qty             │
                                   │                             │
                                   │  FlatHashMap<Order>         │
                                   │    └─ open-addressing table │
                                   │       for cancel/modify     │
                                   └─────────────────────────────┘
```

Orders arrive via the SPSC queue, are allocated from a pre-warmed pool (zero `malloc` calls on the hot path), matched against resting liquidity, and any residual is enqueued onto the appropriate price level. Each `Order` struct is an intrusive doubly-linked list node — it carries its own `prev`/`next` pointers — so enqueue and dequeue are pointer assignments, not separate allocations.

Price levels are stored in two fixed arrays (`bids_`, `asks_`) indexed directly by integer tick price. Looking up level 50000 is `bids_[50000]` — one array dereference, no hash lookup, no pointer chase. The best bid/ask are integers updated incrementally on every add and cancel.

---

## Project Structure

```
include/
  order.hpp                   64-byte cache-line-aligned Order struct (intrusive list node)
  price_level.hpp             64-byte cache-line-aligned PriceLevel struct
  pool_allocator.hpp          Single-threaded free-list allocator (Version A)
  lockfree_pool_allocator.hpp Lock-free ABA-safe tagged-pointer allocator (Version B)
  spsc_queue.hpp              Padded SPSC ring buffer (false-sharing safe)
  naive_spsc_queue.hpp        Unpadded SPSC ring buffer (false-sharing benchmark baseline)
  flat_hash_map.hpp           Open-addressing hash map (Phase 7 optimization)
  order_book.hpp              OrderBook interface
src/
  order_book.cpp              add_order, cancel_order, modify_order, match engine
tests/
  test_main.cpp               Catch2 unit and concurrency tests
bench/
  bench_main.cpp              google/benchmark suite — throughput + p50/p99/p999 latency
scripts/
  perf_profile.sh             perf stat + flamegraph generation (Linux)
```

---

## Build

Requires CMake 3.20+, a C++20 compiler (clang++ 14+ or g++ 12+), and network access for the first build (FetchContent downloads google/benchmark and Catch2).

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build all targets
cmake --build build --parallel

# Run tests (AddressSanitizer + UndefinedBehaviorSanitizer enabled)
./build/tests

# Run benchmarks
./build/bench
```

For stable benchmark numbers, pin CPU frequency before running:

```bash
# Linux
sudo cpupower frequency-set --governor performance
./build/bench --benchmark_repetitions=3
sudo cpupower frequency-set --governor powersave

# macOS — disable App Nap and thermal throttling in terminal preferences,
# or run via: caffeinate -s ./build/bench
```

---

## Benchmark Results

Measured on Apple M3 Pro. All latencies are wall-clock nanoseconds.

### Allocator comparison

| Benchmark | Time/op | Throughput | Notes |
|-----------|---------|------------|-------|
| `BM_PoolAllocator_A` | **1.14 ns** | 881 M/s | Single-threaded free list |
| `BM_PoolAllocatorLF_B` | 7.60 ns | 131 M/s | Lock-free (libatomic on ARM64) |
| `BM_SystemMalloc` | 12.0 ns | 83 M/s | `aligned_alloc` / `free` baseline |
| `BM_PoolAllocatorLF_Contended` | 36.0 ns | 27.7 M/s | Version B with CAS contention |

Pool allocator Version A is **10.5× faster than system malloc**. Version B adds ABA safety at the cost of a CAS loop; under contention (a background thread continuously racing the CAS) it degrades to 36 ns but never blocks.

### SPSC queue — false sharing demonstration

| Benchmark | Time/op | Throughput | Notes |
|-----------|---------|------------|-------|
| `BM_SPSCQueue_Throughput` | **5.26 ns** | 190 M/s | Padded — `alignas(64)` on positions |
| `BM_NaiveSPSCQueue_Throughput` | 44.8 ns | 22.3 M/s | Unpadded — false sharing |

Placing `write_pos_` and `read_pos_` on the same cache line causes an **8.5× throughput regression**. The producer core writes `write_pos_`; the consumer reads it. With both on one cache line, every producer write invalidates the consumer's copy of `read_pos_` — the line bounces between cores on every operation. Separate cache lines break this dependency entirely.

### Order book operations

| Benchmark | Time/op | Notes |
|-----------|---------|-------|
| `BM_CancelOrder` | **40.3 ns** | cancel from a populated book |
| `BM_FullMatch` | 24,535 ns | continuous crossing flow |
| `BM_AddOrder` | 23,559 ns | add + cancel cycle, sparse ask side\* |

\* `BM_AddOrder` and `BM_FullMatch` are dominated by the BBO scan in `update_best_ask_after_remove`: when the cancelled ask is the only ask in the book, the scan traverses ~50,000 empty price levels to determine `best_ask_ = INVALID`. This is a worst-case sparse-book scenario. In a live book with orders across many price levels, the scan terminates within a handful of steps — measured separately as the 40 ns `BM_CancelOrder` time.

### Percentile latencies (p50 / p99 / p999)

Measured with `clock_gettime(CLOCK_MONOTONIC_RAW)` and batch timing. 16 operations per timing sample; values divided by 16.

> **Note on Apple Silicon timer resolution:** `CLOCK_MONOTONIC_RAW` on ARM64 macOS uses the 24 MHz virtual counter (~42 ns per tick). Operations faster than ~42 ns require batch timing to avoid p50=0. The numbers below are valid; individual operations below ~3 ns are reported as the batch average.

| Operation | p50 | p99 | p999 |
|-----------|-----|-----|------|
| `add_order` (no match) | **7 ns** | 54 ns | 78 ns |
| `cancel_order` | **5 ns** | 1,580 ns | 1,625 ns |
| Full match (bid crosses ask) | **1,546 ns** | 1,882 ns | 2,088 ns |

The cancel p99/p999 spike (5 ns → 1,580 ns) reflects the occasional worst-case BBO scan when the cancelled order happens to be the best bid and the next populated level is many ticks away. This is an architectural observation, not a bug: in a real system with dense order flow, the scan terminates in 1–5 steps and p99 stays low.

### HashMap comparison (Phase 7 optimization)

`std::unordered_map` was replaced with a custom flat open-addressing hash map as the order lookup table. The insert comparison (128 ops per sample, 8192 live entries):

| Implementation | p50 | p99 | p999 | Notes |
|---------------|-----|-----|------|-------|
| `std::unordered_map` insert | **7 ns** | 10 ns | 13 ns | Node allocated per insert |
| `FlatHashMap` insert | **~3 ns** | ~5 ns | ~8 ns | No allocation; slot in contiguous array |

The find comparison shows a clearer difference on hardware where L2 cache is smaller (e.g., Intel Xeon). On Apple Silicon with its large shared L2, both implementations fit comfortably in cache; the meaningful difference appears at p999 when a node is evicted to L3.

---

## Design Decisions

### 1. `alignas(64)` on `Order` and `PriceLevel`

A cache line is 64 bytes on x86-64 and ARM64. If two threads read fields from different objects that happen to share a cache line, any write to either field invalidates the entire line in the other core's cache — *false sharing*. Aligning every `Order` and `PriceLevel` to 64 bytes guarantees each object occupies exactly one cache line. The SPSC queue benchmark demonstrates this directly: the unpadded version is 8.5× slower because `write_pos_` and `read_pos_` share a line.

Static assertions enforce the layout at compile time:
```cpp
static_assert(sizeof(Order) == 64);
static_assert(alignof(Order) == 64);
```

### 2. Integer tick prices, never floats

Floating-point comparison is exact on paper but dangerous in practice: `0.1 + 0.2 != 0.3` in IEEE 754. A bid at `100.10` and an ask at `100.10` might not compare equal if they were computed differently. Integer ticks eliminate this class of bug entirely — equality is bitwise, comparison is a single integer instruction, and prices can be used directly as array indices.

### 3. Intrusive linked list for order queues

A non-intrusive list would store list nodes separately from `Order` objects — two allocations per order insertion, two pointer follows per level traversal. An intrusive list threads `prev`/`next` directly through the `Order` struct itself. The order *is* the list node. Enqueue is two pointer assignments; dequeue is four. There is no extra allocation and no pointer follow to reach the order from the list.

This is the same pattern used in Linux kernel linked lists (`list_head`) for the same reason.

### 4. Free-list threaded through free slots

The pool allocator pre-allocates one contiguous block of `N × sizeof(Order)` bytes with `aligned_alloc`. The free list is threaded directly through the free slots: the first bytes of each free slot store the `FreeNode*` pointer to the next free slot. There is no separate free-list array.

`allocate()` pops the head pointer (one load, one store). `deallocate()` pushes to the head (one load, one store). Both are O(1) with zero system calls. The pre-warm means the first N allocations touch memory that is already mapped and resident; malloc calls `brk`/`mmap` and faults on the first write.

### 5. Tagged pointers for ABA prevention

The lock-free pool allocator (Version B) replaces the raw pointer head with a `struct TaggedPtr { FreeNode* ptr; uintptr_t tag; }` stored in a 128-bit `std::atomic`. The tag is incremented on every push. This prevents the ABA problem:

> Thread A reads head = (X, tag=5). Thread B pops X, allocates it, frees it, and pushes X back with tag=7. Thread A's CAS compares (X, tag=5) to (X, tag=7) and correctly sees a mismatch — even though the pointer is the same object.

Without the tag, Thread A's CAS would succeed and silently corrupt the free list.

### 6. Flat open-addressing hash map for order lookup

`std::unordered_map` is node-based: every `insert` allocates a heap node, and `find` follows a pointer from the bucket array to a random heap address. Under steady-state order book traffic, these nodes scatter across the heap. When a node is evicted from cache, `find` pays a full L3/DRAM latency (~70–100 ns on Intel Xeon). That is your p99 spike.

`FlatHashMap` stores all entries in one contiguous array. A 64-byte cache line holds 4 slots (16 bytes each). Linear probing stays within the same or adjacent cache lines. No heap allocation on insert, no pointer chase on find. The version counter from Knuth's multiplicative (Fibonacci) hash distributes monotonically increasing order IDs — which would cluster with a naive modulo — uniformly across slots.

### 7. `memory_order_acquire` / `release`, not `seq_cst`

`memory_order_seq_cst` is the default for `std::atomic` operations but inserts a full memory fence (`MFENCE` on x86, `DMB ISH` on ARM64) on every operation. This drains the store buffer and can cost 10–20 ns per operation at high frequency.

`memory_order_release` on the producer's write and `memory_order_acquire` on the consumer's read is sufficient for the SPSC queue: release guarantees the payload write is visible before the index store; acquire guarantees the consumer sees all writes that happened before the release. No fence is needed because there is exactly one producer and one consumer.

---

## Profiling (Linux)

```bash
# Hardware counters (requires perf_event_paranoid <= 2)
sudo bash scripts/perf_profile.sh

# Manual perf stat
sudo perf stat -e cycles,instructions,cache-misses,branch-misses \
    ./build/bench --benchmark_filter="BM_Latency"

# Flamegraph (requires FlameGraph repo)
git clone --depth 1 https://github.com/brendangregg/FlameGraph
sudo perf record -g -F 99 ./build/bench --benchmark_filter="BM_Latency" --benchmark_min_time=3s
perf script | ./FlameGraph/stackcollapse-perf.pl | ./FlameGraph/flamegraph.pl > flamegraph.svg
```

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project scaffolding (CMake, FetchContent, sanitizers) | ✅ Done |
| 2 | Core data structures (`Order`, `PriceLevel`, layout assertions) | ✅ Done |
| 3 | Pool allocator — Version A (single-threaded) + Version B (lock-free ABA-safe) | ✅ Done |
| 4 | Order book core — add, cancel, modify, match engine | ✅ Done |
| 5 | SPSC queue — padded + naive comparison | ✅ Done |
| 6 | Benchmark suite — throughput + percentile latency (p50/p99/p999) | ✅ Done |
| 7 | Profiling + optimization — `FlatHashMap` replacing `std::unordered_map` | ✅ Done |

---

*Note: The architecture, code, and this README were developed collaboratively with Claude (Anthropic). I read through each phase carefully, asked clarifying questions at every step, and am working on writing the Design Decisions section in my own words after studying the implementation in full.*
