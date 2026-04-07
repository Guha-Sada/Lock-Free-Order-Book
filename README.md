# Lock-Free Order Book

A price-time-priority central limit order book (CLOB) written in C++20, built to explore low-latency systems design. The project includes a custom pool allocator, a lock-free SPSC queue, and a benchmark suite targeting the kind of latency numbers that matter in market microstructure.

## Architecture

Orders are allocated from a pre-warmed pool allocator that never calls `malloc` on the hot path. Each `Order` is a 64-byte cache-line-aligned, intrusive doubly-linked list node — the order *is* the list node, eliminating a separate allocation per level entry. Price levels are stored in two fixed arrays (bids and asks) indexed directly by integer tick price, giving O(1) level lookup with a single array dereference and no pointer chasing. The best bid/ask are tracked as integers updated incrementally on every add and cancel. A single-producer single-consumer ring buffer feeds orders from a producer thread to the book engine without locks or contention.

## Project Structure

```
include/
  order.hpp                  # 64-byte cache-line-aligned Order struct
  price_level.hpp            # 64-byte cache-line-aligned PriceLevel struct
  pool_allocator.hpp         # Single-threaded free-list allocator (Version A)
  lockfree_pool_allocator.hpp # Lock-free tagged-pointer allocator (Version B)
  spsc_queue.hpp             # Padded SPSC ring buffer (false-sharing safe)
  naive_spsc_queue.hpp       # Unpadded SPSC ring buffer (for benchmarking)
  order_book.hpp             # Order book interface
src/
  order_book.cpp             # add_order, cancel_order, modify_order, match
tests/
  test_main.cpp              # Catch2 unit tests
bench/
  bench_main.cpp             # google/benchmark suite
```

## Build

Requires CMake 3.20+ and a C++20 compiler (clang++ or g++).

```bash
# Configure (release build)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build all targets
cmake --build build

# Run tests
./build/tests/order_book_tests

# Run benchmarks
./build/bench/order_book_bench
```

For the most accurate benchmark numbers, disable CPU frequency scaling before running:

```bash
sudo cpupower frequency-set --governor performance
./build/bench/order_book_bench
sudo cpupower frequency-set --governor powersave
```

## Design Decisions

**Integer tick prices, never floats.**

**`alignas(64)` on `Order` and `PriceLevel`.**

**Intrusive linked list for order queues.**

**Pool allocator with free-list threading.**

**Lock-free pool allocator with tagged pointers.**

**SPSC queue with separate cache lines for read/write positions.**

**`memory_order_acquire`/`release`, not `seq_cst`.**

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project scaffolding | Done |
| 2 | Core data structures | Done |
| 3 | Pool allocator (A + B) | Done |
| 4 | Order book core | Done |
| 5 | SPSC queue | Done |
| 6 | Benchmark suite | In progress |
| 7 | Profiling + optimization | Pending |


##work done by claude:

Everything until this point, including the README, except for the Design decisions section. The decisions itself  were made by Claude. After each phase of the plan it created, I read the source code carefully and asked clarifying questions wherever I was confused. So, I add an explanation for these design decisions in my own words after reading the code even more.
