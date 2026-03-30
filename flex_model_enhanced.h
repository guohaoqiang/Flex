#ifndef FLEX_MODEL_ENHANCED_H
#define FLEX_MODEL_ENHANCED_H

#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// H100 SM L1 cache parameters (bytes).
//
// Total L1/shared-memory budget per SM is 256 KB.  With shared memory
// carved out for the kernel, a practical upper bound for the L1 data-
// cache portion is ~192 KB.  We use a conservative 128 KB to leave
// room for instruction cache, texture, and other cache traffic.
// ---------------------------------------------------------------------------
constexpr int H100_L1_CACHE_BYTES = 128 * 1024;  // 128 KB effective L1 data

// ---------------------------------------------------------------------------
// Vectorization: number of B-matrix floats loaded per SASS instruction.
// ---------------------------------------------------------------------------
inline int get_vec_b_size(int kernel_version) {
    switch (kernel_version) {
        case 36: case 38: return 1;   // scalar
        case 37: case 39: return 2;   // float2
        case 40: return 4;            // float4
        default: return 1;
    }
}

// ---------------------------------------------------------------------------
// B-matrix L1 reuse helpers.
//
// Each unique column index in a tile/SM requires loading one row of B
// (K floats = K*4 bytes).  Whether that row stays in L1 depends on
// the working-set size versus L1 capacity.
// ---------------------------------------------------------------------------

// Number of B rows (each K floats) that fit in L1.
inline int l1_b_row_capacity(int K) {
    const int b_row_bytes = K * static_cast<int>(sizeof(float));
    return (b_row_bytes > 0) ? (H100_L1_CACHE_BYTES / b_row_bytes) : 1;
}

// Fraction of the per-SM B-row working set that fits in L1.
//   unique_cols_per_sm – average distinct column indices per SM
//   K                  – dense-matrix width (floats per B row)
// Returns 1.0 when everything fits, <1.0 when L1 is too small.
inline double compute_b_l1_coverage(int unique_cols_per_sm, int K) {
    if (unique_cols_per_sm <= 0) return 1.0;
    const double cap = static_cast<double>(l1_b_row_capacity(K));
    return std::min(1.0, cap / static_cast<double>(unique_cols_per_sm));
}

// Effective B reuse factor blending tile-level and SM-level reuse
// according to how much of the SM's B working set fits in L1.
//
//   nz_p_toc    – NZ per unique tile-column  (tile-level, B-Re1)
//   nz_p_sm_col – NZ per unique SM-column    (SM-level,   B-Re2)
//   l1_coverage – from compute_b_l1_coverage()
//
//   l1_coverage = 1 →  full cross-tile reuse captured by L1 → B-Re2
//   l1_coverage = 0 →  no cross-tile reuse, tile-level only  → B-Re1
inline double compute_effective_b_reuse(
    double nz_p_toc, double nz_p_sm_col, double l1_coverage) {
    return l1_coverage * nz_p_sm_col + (1.0 - l1_coverage) * nz_p_toc;
}

// ---------------------------------------------------------------------------
// Per-NZ load-transaction estimates (for the flex-vs-work-queue decision
// in mat.cu).
//
// Returns estimated L2/HBM load transactions attributable to ONE non-zero.
// Both paths model the same quantities so they can be compared directly.
//
// To convert to "loads per MADD" (matching the G metric), divide by K.
// ---------------------------------------------------------------------------

// Flex (alpha-tile) path – loads per NZ.
//
//   col_reuse      – times this column appears in the tile
//   need_atomic    – whether the row requires an atomic C update
//   kernel_version – for vectorization width
//   K              – B-matrix width (feature dimension)
//   n_unique_cols  – unique column indices in the tile (for L1 sizing)
inline double estimate_flex_ld_per_nz(
    int col_reuse,
    bool need_atomic,
    int kernel_version,
    int K,
    int n_unique_cols) {

    const int vec_b = get_vec_b_size(kernel_version);

    // Weight loads: column index + value = 2 (always from L2/HBM).
    const double weight_ld = 2.0;

    // B loads: K / vec_b load insns for one full B row, amortised across
    // col_reuse NZ sharing the column in this tile.
    const double b_ld_per_nz_tile =
        static_cast<double>(K) / vec_b / std::max(1, col_reuse);

    // L1 coverage: what fraction of this tile's B rows fit in L1?
    // Rows that fit get served from L1 after the cold miss, so only
    // the cold-miss fraction generates L2/HBM traffic.
    const double l1_cov = compute_b_l1_coverage(n_unique_cols, K);
    // Miss fraction ≈ 1 for unique accesses (cold miss), reduced by
    // l1_cov for the reused accesses within the tile.
    const double miss_frac =
        1.0 - l1_cov * (1.0 - 1.0 / std::max(1, col_reuse));
    const double b_ld = b_ld_per_nz_tile * miss_frac;

    // Atomic C update (one read-modify-write per element if needed).
    const double atomic_ld = need_atomic ? 1.0 : 0.0;

    return weight_ld + b_ld + atomic_ld;
}

// Work-queue path – loads per NZ.
//   No tiling reuse for B: every NZ loads its own full B row.
inline double estimate_work_ld_per_nz(
    bool need_atomic,
    int kernel_version,
    int K) {

    const int vec_b = get_vec_b_size(kernel_version);

    const double weight_ld = 2.0;
    const double b_ld = static_cast<double>(K) / vec_b; // no column reuse
    const double atomic_ld = need_atomic ? 1.0 : 0.0;

    // Work-queue index / counter overhead.
    const double overhead = 4.0;

    return weight_ld + b_ld + atomic_ld + overhead;
}

#endif  // FLEX_MODEL_ENHANCED_H
