cmake_minimum_required(VERSION 3.20)
project(orderbook CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generates compile_commands.json for clangd / LSP support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Dependencies (fetched at configure time, no manual install needed)
# ---------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.4
    GIT_SHALLOW    TRUE
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.6.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

# ---------------------------------------------------------------------------
# Shared compile warnings (applied everywhere)
# ---------------------------------------------------------------------------
set(WARN_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
)

# ---------------------------------------------------------------------------
# Core library (header-heavy; order_book.cpp holds the bulk of the logic)
# ---------------------------------------------------------------------------
add_library(orderbook_lib STATIC
    src/order_book.cpp
)
target_include_directories(orderbook_lib PUBLIC include)
target_compile_options(orderbook_lib PRIVATE ${WARN_FLAGS})

# ---------------------------------------------------------------------------
# Release library: maximum optimisation, used by benchmarks
# ---------------------------------------------------------------------------
add_library(orderbook_lib_release STATIC
    src/order_book.cpp
)
target_include_directories(orderbook_lib_release PUBLIC include)
target_compile_options(orderbook_lib_release PRIVATE
    ${WARN_FLAGS}
    -O3
    -march=native
    # -mcx16 enables CMPXCHG16B on x86-64 for hardware lock-free 128-bit atomics.
    # Uncomment on x86-64 Linux/Windows.  On ARM64 (Apple Silicon, Graviton),
    # Clang handles 128-bit atomics natively; GCC requires -latomic (see below).
    -DNDEBUG
    -fno-omit-frame-pointer   # keeps perf/flamegraph stack traces readable
)

# ---------------------------------------------------------------------------
# Test executable: debug flags + ASan/UBSan
# ---------------------------------------------------------------------------
add_executable(tests
    tests/test_main.cpp
)
target_link_libraries(tests PRIVATE orderbook_lib Catch2::Catch2WithMain)
target_compile_options(tests PRIVATE
    ${WARN_FLAGS}
    -g
    -O1
    -fsanitize=address,undefined
)
target_link_options(tests PRIVATE -fsanitize=address,undefined)
# -latomic: required on GCC/ARM64 for 128-bit std::atomic (LockFreePoolAllocator).
# On x86-64 GCC with -mcx16, or Clang on any platform, this is not needed.
target_link_libraries(tests PRIVATE atomic)

# ---------------------------------------------------------------------------
# Benchmark executable: release flags
# ---------------------------------------------------------------------------
add_executable(bench
    bench/bench_main.cpp
)
target_link_libraries(bench PRIVATE orderbook_lib_release benchmark::benchmark benchmark::benchmark_main atomic)
target_compile_options(bench PRIVATE
    -O3
    -march=native
    -mcx16
    -DNDEBUG
    -fno-omit-frame-pointer
)
