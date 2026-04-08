#!/usr/bin/env bash
# =============================================================================
# perf_profile.sh — Phase 7 profiling driver
#
# USAGE (Linux only, requires perf installed):
#   cd <project root>
#   cmake --build build --target bench
#   sudo bash scripts/perf_profile.sh
#
# On macOS use Instruments instead:
#   instruments -t "Time Profiler" ./build/bench \
#       --benchmark_filter="BM_Latency" \
#       --benchmark_min_time=3s
#
# What each section does:
#
#   perf stat   — hardware counters for one benchmark run.
#                 Key numbers to watch:
#                   - cache-misses / cache-references  → LLC miss rate
#                   - branch-misses                    → misprediction rate
#                   - instructions / cycles            → IPC (higher = better)
#
#   perf record — samples the call stack at 99 Hz during the run.
#                 The -g flag enables frame-pointer-based stack unwinding.
#                 Outputs perf.data.
#
#   perf script → stackcollapse-perf.pl → flamegraph.pl
#               — converts perf.data into a flamegraph SVG.
#                 Requires Brendan Gregg's FlameGraph scripts:
#                   git clone https://github.com/brendangregg/FlameGraph
#
# =============================================================================
set -euo pipefail

BENCH="./build/bench"
FILTER="BM_Latency"        # profile only the latency benchmarks
MIN_TIME="3"               # seconds per benchmark (more samples = better flame)
FLAMEGRAPH_DIR="./FlameGraph"
OUT_DIR="./profiles"

mkdir -p "$OUT_DIR"

if [[ ! -x "$BENCH" ]]; then
    echo "ERROR: $BENCH not found. Run: cmake --build build --target bench"
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Hardware counter summary (perf stat)
# ---------------------------------------------------------------------------
echo "=== perf stat ==="
perf stat \
    -e cycles,instructions,cache-misses,cache-references,branch-misses,branch-instructions \
    "$BENCH" \
    --benchmark_filter="$FILTER" \
    --benchmark_min_time="${MIN_TIME}s" \
    2>&1 | tee "$OUT_DIR/perf_stat.txt"

# ---------------------------------------------------------------------------
# 2. Call-graph sampling (perf record)
# ---------------------------------------------------------------------------
echo ""
echo "=== perf record ==="
perf record \
    -g \
    --call-graph fp \
    -F 99 \
    -o "$OUT_DIR/perf.data" \
    -- "$BENCH" \
    --benchmark_filter="$FILTER" \
    --benchmark_min_time="${MIN_TIME}s"

# ---------------------------------------------------------------------------
# 3. Flamegraph generation
#    Clone FlameGraph repo if not present.
# ---------------------------------------------------------------------------
if [[ ! -d "$FLAMEGRAPH_DIR" ]]; then
    echo ""
    echo "=== Cloning FlameGraph scripts ==="
    git clone --depth 1 https://github.com/brendangregg/FlameGraph "$FLAMEGRAPH_DIR"
fi

echo ""
echo "=== Generating flamegraph ==="
perf script -i "$OUT_DIR/perf.data" \
    | "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" \
    | "$FLAMEGRAPH_DIR/flamegraph.pl" \
        --title "OrderBook Latency Flamegraph" \
        --width 1800 \
    > "$OUT_DIR/flamegraph.svg"

echo ""
echo "Flamegraph written to: $OUT_DIR/flamegraph.svg"
echo "Open it in a browser: open $OUT_DIR/flamegraph.svg"
