#ifndef FLEX_MODEL_ENHANCED_H
#define FLEX_MODEL_ENHANCED_H

#include <cmath>

// Vectorization efficiency (relative to scalar baseline)
inline double get_vectorization_efficiency(int kernel_version) {
    switch (kernel_version) {
        case 36: case 38: return 1.0;   // scalar
        case 37: case 39: return 0.5;   // float2 (50% traffic)
        case 40: return 0.25;            // float4 (25% traffic)
        default: return 1.0;
    }
}

// L2 cache penalty coefficients (kernel-dependent)
inline double get_l2_penalty(int kernel_version) {
    switch (kernel_version) {
        case 36: case 38: return 0.08;  // scalar
        case 37: case 39: return 0.04;  // float2
        case 40: return 0.02;            // float4
        default: return 0.08;
    }
}

// Instruction overhead coefficients
inline double get_instr_overhead(int kernel_version) {
    switch (kernel_version) {
        case 36: case 38: return 0.10;
        case 37: case 39: return 0.05;
        case 40: return 0.025;
        default: return 0.10;
    }
}

// L1 reuse factor based on column reuse count
// exp(-u/2.5) where u = times column accessed in tile
inline double compute_l1_reuse_factor(int col_reuse_count) {
    const double decay = 2.5;
    if (col_reuse_count <= 0) return 1.0;
    return std::exp(-static_cast<double>(col_reuse_count) / decay);
}

// Enhanced cost function for flex tiles
// Returns cost value (lower = prefer flex, higher = prefer work queue)
inline double compute_flex_cost_enhanced(
    int nnz, 
    int col_reuse_count, 
    bool need_atomic, 
    int kernel_version, 
    int tile_rows) {
    
    const double C_HBM = 1.0;      // HBM cost (normalized)
    const double C_COMP = 0.5;     // Computation cost
    const double C_ATOMIC = 50.0;  // Atomic operation cost
    
    // Get kernel-specific parameters
    double C_L2 = get_l2_penalty(kernel_version);
    double C_INSTR = get_instr_overhead(kernel_version);
    double vect_eff = get_vectorization_efficiency(kernel_version);
    
    // L1 reuse benefit: exp(-u/2.5)
    double l1_reuse_factor = compute_l1_reuse_factor(col_reuse_count);
    
    // Atomic cost (if needed)
    double atomic_cost = need_atomic ? C_ATOMIC : 0.0;
    
    // Total flex cost
    double flex_cost = 
        nnz * (C_HBM * vect_eff + C_L2 * l1_reuse_factor + C_INSTR) + 
        C_COMP + 
        atomic_cost;
    
    return flex_cost;
}

// Work queue cost function (baseline for comparison)
inline double compute_work_cost(int nnz, bool need_atomic) {
    const double C_ENQUEUE = 100.0;      // Enqueue overhead
    const double C_PROCESS = 2.0;        // Per-nnz processing
    const double C_ATOMIC_WQ = 75.0;     // Atomic in work queue
    
    double atomic_cost = need_atomic ? C_ATOMIC_WQ : 0.0;
    return C_ENQUEUE + nnz * C_PROCESS + atomic_cost;
}

#endif  // FLEX_MODEL_ENHANCED_H
