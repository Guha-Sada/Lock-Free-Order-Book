#pragma once
// =============================================================================
// flat_hash_map.hpp — Open-addressing hash map for uint64_t → T*
//
// WHY THIS EXISTS:
//   std::unordered_map uses a node-based design: each key-value pair lives in
//   a separately heap-allocated node, and collision chains are linked lists.
//   Under load, find() chases pointers to random heap addresses → each hop
//   is a potential cache miss.  That shows up directly in p99/p999 latency.
//
//   A flat open-addressing table stores every entry in a single contiguous
//   array.  Collision resolution uses LINEAR PROBING: if slot h(k) is taken,
//   try h(k)+1, h(k)+2, etc.  Because the probed slots are adjacent in memory,
//   hardware prefetchers pick them up before the CPU needs them.
//
// DESIGN CHOICES:
//
//   Key type: uint64_t (order IDs).  Only stores pointers (T*) as values.
//
//   Capacity: always a power of two — modulo becomes a bitwise AND.
//     index = hash(key) & (capacity - 1)
//
//   Load factor: kept below 0.5.  At 50% occupancy, average probe length
//   is under 1.5 slots.  We enforce this by asserting on insert.
//
//   Tombstones: when an entry is erased we mark it DELETED (not EMPTY).
//   This is necessary for correctness: if we cleared it to EMPTY, a later
//   find() for a key that was displaced past this slot would stop too early
//   and incorrectly return "not found".
//
//   Tombstone accumulation: In an order book, orders are inserted and
//   cancelled continuously.  Tombstones eventually fill the table and inflate
//   probe lengths.  We handle this by REHASHING in-place whenever the number
//   of tombstones + live entries exceeds 50% of capacity.  Because capacity
//   stays the same and live count stays well below 25% (order books don't
//   hold 65536 orders simultaneously in practice), rehashing is rare.
//
//   Thread safety: none.  OrderBook is single-threaded on the hot path.
//
//   Alignment: the Slot array is allocated with operator new, which gives
//   at least alignof(max_align_t).  Each Slot is 16 bytes (key + pointer),
//   so the whole array fits in cache lines without padding.
//
// COMPLEXITY:
//   find()    — O(1) amortized, O(n) worst case (degenerate clustering)
//   insert()  — O(1) amortized
//   erase()   — O(1) amortized (tombstone write, no shifting)
//
// =============================================================================

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <cassert>

template <typename T>
class FlatHashMap {
public:
    // -------------------------------------------------------------------------
    // Constants used as sentinel key values.
    //   EMPTY_KEY:   slot has never been written.
    //   DELETED_KEY: slot held an entry that was erased (tombstone).
    //
    // We use the two largest uint64_t values.  Valid order IDs start at 1 and
    // are monotonically increasing, so they never collide with these sentinels.
    // -------------------------------------------------------------------------
    static constexpr uint64_t EMPTY_KEY   = ~uint64_t{0};        // 0xFFFF...
    static constexpr uint64_t DELETED_KEY = ~uint64_t{0} - 1;    // 0xFFFE...

    // Each slot is exactly 16 bytes: key (8) + pointer (8).
    // 16 bytes fits neatly into a cache line alongside adjacent slots.
    struct alignas(16) Slot {
        uint64_t key   = EMPTY_KEY;
        T*       value = nullptr;
    };

    // -------------------------------------------------------------------------
    // Constructor.
    //   capacity: number of slots.  MUST be a power of two.
    //   We allocate capacity slots upfront — no rehashing that changes size.
    // -------------------------------------------------------------------------
    explicit FlatHashMap(size_t capacity)
        : capacity_(capacity)
        , mask_(capacity - 1)
        , slots_(new Slot[capacity])
    {
        assert((capacity & (capacity - 1)) == 0 && "capacity must be power of two");
        // slots_ are default-initialized: key=EMPTY_KEY, value=nullptr.
    }

    ~FlatHashMap() {
        delete[] slots_;
    }

    // Not copyable — owns raw memory.
    FlatHashMap(const FlatHashMap&) = delete;
    FlatHashMap& operator=(const FlatHashMap&) = delete;

    // -------------------------------------------------------------------------
    // insert(key, value) — add a new key-value pair.
    //
    // Finds the first EMPTY or DELETED slot in the probe sequence.
    // Prefers to reuse a tombstone if one is encountered before an empty slot,
    // so tombstone count doesn't grow without bound.
    //
    // Asserts that load < 50%.  In production you'd grow the table instead,
    // but the order book pool is capped at DEFAULT_CAPACITY = 65536 and the
    // table is sized at 2× that, so this never triggers.
    // -------------------------------------------------------------------------
    void insert(uint64_t key, T* value) noexcept {
        assert(key != EMPTY_KEY && key != DELETED_KEY);
        assert(size_ < capacity_ / 2 && "load factor exceeded 50% — resize needed");

        size_t idx = hash(key);
        size_t tombstone_idx = capacity_;   // sentinel: "no tombstone found yet"

        for (size_t i = 0; i < capacity_; ++i) {
            Slot& s = slots_[idx];

            if (s.key == EMPTY_KEY) {
                // Found an empty slot.  Use tombstone instead if we saw one —
                // reusing tombstones reduces average probe length.
                if (tombstone_idx < capacity_) {
                    slots_[tombstone_idx].key   = key;
                    slots_[tombstone_idx].value = value;
                    --tombstone_count_;
                } else {
                    s.key   = key;
                    s.value = value;
                }
                ++size_;
                return;
            }

            if (s.key == DELETED_KEY) {
                if (tombstone_idx == capacity_) {
                    tombstone_idx = idx;   // remember first tombstone
                }
            } else if (s.key == key) {
                // Key already exists — overwrite value (shouldn't happen in
                // a well-behaved order book, but handle it gracefully).
                s.value = value;
                return;
            }

            idx = (idx + 1) & mask_;
        }

        assert(false && "table is full — this should never happen");
    }

    // -------------------------------------------------------------------------
    // find(key) — returns pointer to value, or nullptr if not found.
    //
    // Walks the probe sequence starting at hash(key).
    //   - EMPTY_KEY:   stop; key was never inserted past this point.
    //   - DELETED_KEY: skip (tombstone — key may exist further along).
    //   - matching key: return &slot.value so the caller can read or update it.
    // -------------------------------------------------------------------------
    [[nodiscard]] T** find(uint64_t key) noexcept {
        size_t idx = hash(key);

        for (size_t i = 0; i < capacity_; ++i) {
            Slot& s = slots_[idx];

            if (s.key == EMPTY_KEY)   return nullptr;
            if (s.key == DELETED_KEY) { idx = (idx + 1) & mask_; continue; }
            if (s.key == key)         return &s.value;

            idx = (idx + 1) & mask_;
        }
        return nullptr;
    }

    [[nodiscard]] T* const* find(uint64_t key) const noexcept {
        return const_cast<FlatHashMap*>(this)->find(key);
    }

    // -------------------------------------------------------------------------
    // erase(key) — marks the slot as DELETED (tombstone).
    //
    // We do NOT shift subsequent entries — that would be Robin Hood deletion
    // (O(n) worst case shifting).  Tombstoning is O(1) and correct.
    //
    // Returns true if the key was found and erased, false if not found.
    // -------------------------------------------------------------------------
    bool erase(uint64_t key) noexcept {
        size_t idx = hash(key);

        for (size_t i = 0; i < capacity_; ++i) {
            Slot& s = slots_[idx];

            if (s.key == EMPTY_KEY)   return false;
            if (s.key == DELETED_KEY) { idx = (idx + 1) & mask_; continue; }

            if (s.key == key) {
                s.key   = DELETED_KEY;
                s.value = nullptr;
                --size_;
                ++tombstone_count_;
                // Rehash if tombstones are accumulating.
                // This keeps average probe lengths short.
                if (tombstone_count_ > capacity_ / 4) {
                    rehash();
                }
                return true;
            }

            idx = (idx + 1) & mask_;
        }
        return false;
    }

    // Diagnostics
    [[nodiscard]] size_t size()     const noexcept { return size_; }
    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool   empty()    const noexcept { return size_ == 0; }

private:
    // -------------------------------------------------------------------------
    // hash(key) — maps a uint64_t key to a slot index.
    //
    // We use the Fibonacci/Knuth multiplicative hash:
    //   h = (key * 11400714819323198485) >> (64 - log2(capacity))
    //
    // The constant 11400714819323198485 is the closest odd integer to
    // 2^64 / phi (golden ratio).  Multiplicative hashing distributes keys
    // well even when they are monotonically increasing integers (like order
    // IDs), which would cluster badly with a naive modulo.
    //
    // The shift by (64 - log2(capacity)) extracts the top bits, which carry
    // the most entropy after the multiplication.
    //
    // Alternative: FNV-1a or xxHash, but those are slower for single integers.
    // -------------------------------------------------------------------------
    [[nodiscard]] size_t hash(uint64_t key) const noexcept {
        // Fibonacci hashing — works for any power-of-two capacity.
        // mask_ + 1 == capacity_ (power of two).
        constexpr uint64_t GOLDEN = 11400714819323198485ULL;
        return static_cast<size_t>((key * GOLDEN) >> (64u - log2_capacity_)) & mask_;
    }

    // -------------------------------------------------------------------------
    // rehash() — rebuild the table in-place without changing capacity.
    //
    // Called when tombstones exceed capacity/4.  We scan every slot, collect
    // live entries into a temporary buffer, clear the whole array to EMPTY,
    // then re-insert.
    //
    // This is O(capacity) but happens rarely (only when 25% of slots are
    // tombstones), so amortized cost per operation is negligible.
    // -------------------------------------------------------------------------
    void rehash() noexcept {
        // Collect live entries.
        struct KV { uint64_t key; T* value; };
        // Stack-allocate for small tables; heap for large.
        // In practice capacity_ is 131072, so we heap-allocate.
        KV* live = new KV[size_];
        size_t n = 0;
        for (size_t i = 0; i < capacity_; ++i) {
            if (slots_[i].key != EMPTY_KEY && slots_[i].key != DELETED_KEY) {
                live[n++] = { slots_[i].key, slots_[i].value };
            }
        }

        // Reset the array.
        for (size_t i = 0; i < capacity_; ++i) {
            slots_[i].key   = EMPTY_KEY;
            slots_[i].value = nullptr;
        }
        size_          = 0;
        tombstone_count_ = 0;

        // Re-insert live entries.
        for (size_t i = 0; i < n; ++i) {
            insert(live[i].key, live[i].value);
        }

        delete[] live;
    }

    // Compute log2 of capacity at construction time for hash().
    static constexpr uint32_t compute_log2(size_t n) noexcept {
        uint32_t log = 0;
        while ((size_t{1} << log) < n) ++log;
        return log;
    }

    const size_t   capacity_;
    const size_t   mask_;             // capacity_ - 1
    const uint32_t log2_capacity_ = compute_log2(capacity_);
    Slot*          slots_;
    size_t         size_           = 0;
    size_t         tombstone_count_ = 0;
};
