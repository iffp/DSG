/**
 * Range Loop Benchmark
 * --------------------
 * Compares two single-threaded (non-SIMD) interval membership checks:
 * 1. Legacy path that stores each interval inside a `CompressedPoint` with a
 *    tuple of endpoints and repeatedly calls `if_in_compressed_range`.
 * 2. Optimized path that keeps endpoints in a structure-of-arrays layout but
 *    still evaluates both the left and right bounds every iteration to mirror
 *    the true range loop, so any speedup comes purely from memory layout and
 *    branching improvements instead of caching the filter result.
 *
 * Exactly `candidate_ratio` of the generated ranges will contain the query's
 * left endpoint so both implementations face the same workload each iteration.
 *
 * Both variants therefore remain O(num_iterations * num_ranges) time and
 * O(num_ranges) space, but the structure-of-arrays version has far better
 * constants thanks to contiguous access and early-continue branching.
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

struct CompressedPoint {
    CompressedPoint(unsigned ext_id, unsigned ll, unsigned lr, unsigned rl, unsigned rr)
        : external_id(ext_id), bounds(std::make_tuple(ll, lr, rl, rr)) {}

    bool if_in_compressed_range(unsigned query_L, unsigned query_R) const {
        const auto &[ll, lr, rl, rr] = bounds;
        return (ll <= query_L && query_L <= lr) && (rl <= query_R && query_R <= rr);
    }

    unsigned external_id;
    std::tuple<unsigned, unsigned, unsigned, unsigned> bounds;
};

int main() {
    constexpr size_t num_iterations = 1;
    constexpr size_t num_ranges = 5000;
    constexpr double candidate_ratio = 0.2; // target 20% of ranges satisfying left filter

    std::mt19937 rng(0);
    std::uniform_int_distribution<unsigned> range_dist(0, 1000);

    const unsigned query_L = range_dist(rng);
    const unsigned query_R = range_dist(rng);

    const size_t target_candidates = std::max<size_t>(1, static_cast<size_t>(num_ranges * candidate_ratio));
    std::vector<uint8_t> candidate_mask(num_ranges, 0);
    for (size_t i = 0; i < target_candidates && i < num_ranges; ++i) {
        candidate_mask[i] = 1;
    }
    std::shuffle(candidate_mask.begin(), candidate_mask.end(), rng);

    std::vector<unsigned> left_lower(num_ranges);
    std::vector<unsigned> left_upper(num_ranges);
    std::vector<unsigned> right_lower(num_ranges);
    std::vector<unsigned> right_upper(num_ranges);
    std::vector<size_t> filtered_indices;
    filtered_indices.reserve(target_candidates);
    std::vector<CompressedPoint> compressed_points;
    compressed_points.reserve(num_ranges);

    auto clamp = []( unsigned value, unsigned min_value, unsigned max_value ) {
        return std::min(max_value, std::max(min_value, value));
    };

    for (size_t i = 0; i < num_ranges; ++i) {
        const bool make_candidate = candidate_mask[i] != 0;
        unsigned ll = 0;
        unsigned lr = 0;
        if (make_candidate) {
            const unsigned left_span = std::min<unsigned>(query_L, rng() % 25);
            ll = query_L - left_span;
            const unsigned right_span = rng() % 25 + 1;
            lr = clamp(query_L + right_span, query_L, 1000u);
        } else {
            if (query_L > 0 && (rng() & 1u)) {
                const unsigned offset = std::min<unsigned>(query_L, 1u + (rng() % 50));
                lr = query_L - offset;
                const unsigned span = std::min<unsigned>(lr, rng() % 50);
                ll = (lr >= span) ? lr - span : 0u;
            } else {
                const unsigned offset = 1u + (rng() % 50);
                ll = clamp(query_L + offset, query_L + 1, 1000u);
                const unsigned span = rng() % 50;
                lr = clamp(ll + span, ll, 1000u);
            }
        }

        const unsigned rl = range_dist(rng);
        const unsigned rr = rl + (rng() % 50);

        left_lower[i] = ll;
        left_upper[i] = lr;
        right_lower[i] = rl;
        right_upper[i] = rr;
        compressed_points.emplace_back(static_cast<unsigned>(i), ll, lr, rl, rr);

        if (query_L >= ll && query_L <= lr) {
            filtered_indices.push_back(i);
        }
    }

    using clock = std::chrono::high_resolution_clock;

    // Legacy CompressedPoint scan
    auto start = clock::now();
    unsigned legacy_answer = 0;
    for (size_t iter = 0; iter < num_iterations; ++iter) {
        for (const auto &point : compressed_points) {
            if (point.if_in_compressed_range(query_L, query_R)) {
                legacy_answer = static_cast<unsigned>(iter);
            }
        }
    }
    auto end = clock::now();
    std::chrono::duration<double> legacy_elapsed = end - start;

    std::cout << "legacy result: " << legacy_answer << '\n';
    std::cout << "CompressedPoint scan " << num_iterations << " times over " << num_ranges
              << " ranges took: " << legacy_elapsed.count() << " seconds\n";

    // Optimized structure-of-arrays scan that evaluates both bounds every iteration
    start = clock::now();
    unsigned optimized_answer = 0;
    const unsigned *const ll_ptr = left_lower.data();
    const unsigned *const lr_ptr = left_upper.data();
    const unsigned *const rl_ptr = right_lower.data();
    const unsigned *const rr_ptr = right_upper.data();
    
    for (size_t iter = 0; iter < num_iterations; ++iter) {
        for (size_t idx = 0; idx < num_ranges; ++idx) {
            const unsigned ll = ll_ptr[idx];
            const unsigned lr = lr_ptr[idx];
            if (query_L < ll || query_L > lr) {
                continue;
            }

            const unsigned rl = rl_ptr[idx];
            const unsigned rr = rr_ptr[idx];
            if (query_R >= rl && query_R <= rr) {
                optimized_answer = static_cast<unsigned>(iter);
            }
        }
    }
    end = clock::now();
    std::chrono::duration<double> optimized_elapsed = end - start;

    std::cout << "optimized result: " << optimized_answer << '\n';
    std::cout << "SoA scan " << num_iterations << " times over " << num_ranges
              << " ranges took: " << optimized_elapsed.count() << " seconds\n";

    const double actual_ratio = static_cast<double>(filtered_indices.size()) / static_cast<double>(num_ranges);
    std::cout << "Candidate ratio: " << actual_ratio * 100.0 << "%\n";
    std::cout << "Filtered indices (" << filtered_indices.size() << "):";
    std::cout << '\n';

    return 0;
}
