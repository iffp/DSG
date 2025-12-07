/**
 * SIMD Range Loop Benchmark
 * -------------------------
 * Measures how much AVX2/AVX512 vectorization can accelerate the structure-of-
 * arrays range containment scan. All intervals are stored in contiguous float
 * arrays, and both the scalar and SIMD kernels perform the same
 * O(num_iterations * num_ranges) work so their timings are directly comparable.
 *
 * Time:  O(num_iterations * num_ranges)
 * Space: O(num_ranges) for the four endpoint arrays.
 */

#include <immintrin.h>
#include <algorithm>

#include <chrono>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

struct ScanResult {
    unsigned last_hit;
    double seconds;
    std::vector<size_t> filtered_indices;
};

ScanResult run_scalar(const std::vector<float> &left_lower,
                      const std::vector<float> &left_upper,
                      const std::vector<float> &right_lower,
                      const std::vector<float> &right_upper,
                      unsigned query_L,
                      unsigned query_R) {
    const size_t n = left_lower.size();
    unsigned answer = 0;
    std::vector<size_t> filtered_indices;
    filtered_indices.reserve(n);
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t idx = 0; idx < n; ++idx) {
        const float ll = left_lower[idx];
        const float lr = left_upper[idx];
        if (query_L < ll || query_L > lr) {
            continue;
        }
        const float rl = right_lower[idx];
        const float rr = right_upper[idx];
        if (query_R >= rl && query_R <= rr) {
            answer = 0;
            filtered_indices.push_back(idx);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    return {answer, elapsed.count(), std::move(filtered_indices)};
}

namespace {

ScanResult run_simd_avx2(const std::vector<float> &left_lower,
                         const std::vector<float> &left_upper,
                         const std::vector<float> &right_lower,
                         const std::vector<float> &right_upper,
                         unsigned query_L,
                         unsigned query_R) {
    const size_t n = left_lower.size();
    const float *ll_ptr = left_lower.data();
    const float *lr_ptr = left_upper.data();
    const float *rl_ptr = right_lower.data();
    const float *rr_ptr = right_upper.data();

    const __m256 qL_vec = _mm256_set1_ps(static_cast<float>(query_L));
    const __m256 qR_vec = _mm256_set1_ps(static_cast<float>(query_R));

    unsigned answer = 0;
    std::vector<size_t> filtered_indices;
    filtered_indices.reserve(n);
    auto start = std::chrono::high_resolution_clock::now();
    size_t idx = 0;
    const size_t vec_limit = n - (n % 8);
    for (; idx < vec_limit; idx += 8) {
        const __m256 ll = _mm256_loadu_ps(ll_ptr + idx);
        const __m256 lr = _mm256_loadu_ps(lr_ptr + idx);
        const __m256 rl = _mm256_loadu_ps(rl_ptr + idx);
        const __m256 rr = _mm256_loadu_ps(rr_ptr + idx);

        const __m256 left_ok = _mm256_and_ps(
            _mm256_cmp_ps(ll, qL_vec, _CMP_LE_OQ),
            _mm256_cmp_ps(qL_vec, lr, _CMP_LE_OQ));

        const __m256 right_ok = _mm256_and_ps(
            _mm256_cmp_ps(rl, qR_vec, _CMP_LE_OQ),
            _mm256_cmp_ps(qR_vec, rr, _CMP_LE_OQ));
        const __m256 both_ok = _mm256_and_ps(left_ok, right_ok);
        int mask = _mm256_movemask_ps(both_ok);
        if (mask) {
            answer = 0;
            while (mask) {
                const int bit = __builtin_ctz(mask);
                filtered_indices.push_back(idx + static_cast<size_t>(bit));
                mask &= mask - 1;
            }
        }
    }

    for (; idx < n; ++idx) {
        const float ll = ll_ptr[idx];
        const float lr = lr_ptr[idx];
        if (query_L < ll || query_L > lr) {
            continue;
        }
        const float rl = rl_ptr[idx];
        const float rr = rr_ptr[idx];
        if (query_R >= rl && query_R <= rr) {
            answer = 0;
            filtered_indices.push_back(idx);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    return {answer, elapsed.count(), std::move(filtered_indices)};
}

#if defined(__AVX512F__)
ScanResult run_simd_avx512(const std::vector<float> &left_lower,
                           const std::vector<float> &left_upper,
                           const std::vector<float> &right_lower,
                           const std::vector<float> &right_upper,
                           unsigned query_L,
                           unsigned query_R) {
    const size_t n = left_lower.size();
    const float *ll_ptr = left_lower.data();
    const float *lr_ptr = left_upper.data();
    const float *rl_ptr = right_lower.data();
    const float *rr_ptr = right_upper.data();

    const __m512 qL_vec = _mm512_set1_ps(static_cast<float>(query_L));
    const __m512 qR_vec = _mm512_set1_ps(static_cast<float>(query_R));

    unsigned answer = 0;
    std::vector<size_t> filtered_indices;
    filtered_indices.reserve(n);
    auto start = std::chrono::high_resolution_clock::now();
    size_t idx = 0;
    const size_t vec_limit = n - (n % 16);
    for (; idx < vec_limit; idx += 16) {
        const __m512 ll = _mm512_loadu_ps(ll_ptr + idx);
        const __m512 lr = _mm512_loadu_ps(lr_ptr + idx);
        const __m512 rl = _mm512_loadu_ps(rl_ptr + idx);
        const __m512 rr = _mm512_loadu_ps(rr_ptr + idx);

        const __mmask16 left_ok = _mm512_kand(
            _mm512_cmp_ps_mask(ll, qL_vec, _CMP_LE_OQ),
            _mm512_cmp_ps_mask(qL_vec, lr, _CMP_LE_OQ));
        const __mmask16 right_ok = _mm512_kand(
            _mm512_cmp_ps_mask(rl, qR_vec, _CMP_LE_OQ),
            _mm512_cmp_ps_mask(qR_vec, rr, _CMP_LE_OQ));
        unsigned mask = static_cast<unsigned>(_mm512_kand(left_ok, right_ok));
        if (mask) {
            answer = 0;
            while (mask) {
                const unsigned bit = static_cast<unsigned>(__builtin_ctz(mask));
                filtered_indices.push_back(idx + static_cast<size_t>(bit));
                mask &= mask - 1;
            }
        }
    }

    for (; idx < n; ++idx) {
        const float ll = ll_ptr[idx];
        const float lr = lr_ptr[idx];
        if (query_L < ll || query_L > lr) {
            continue;
        }
        const float rl = rl_ptr[idx];
        const float rr = rr_ptr[idx];
        if (query_R >= rl && query_R <= rr) {
            answer = 0;
            filtered_indices.push_back(idx);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    return {answer, elapsed.count(), std::move(filtered_indices)};
}
#endif  // __AVX512F__

}  // namespace

ScanResult run_simd(const std::vector<float> &left_lower,
                    const std::vector<float> &left_upper,
                    const std::vector<float> &right_lower,
                    const std::vector<float> &right_upper,
                    unsigned query_L,
                    unsigned query_R) {
#if defined(__AVX512F__)
    return run_simd_avx512(left_lower, left_upper, right_lower, right_upper,
                           query_L, query_R);
#else
    return run_simd_avx2(left_lower, left_upper, right_lower, right_upper,
                         query_L, query_R);
#endif
}

constexpr const char *simd_backend_name() {
#if defined(__AVX512F__)
    return "AVX-512";
#else
    return "AVX2";
#endif
}

int main() {
    constexpr size_t num_ranges = 5000;
    constexpr double candidate_ratio = 0.2;

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

    auto clamp = [](unsigned value, unsigned min_value, unsigned max_value) {
        return std::min(max_value, std::max(min_value, value));
    };

    std::vector<float> left_lower(num_ranges);
    std::vector<float> left_upper(num_ranges);
    std::vector<float> right_lower(num_ranges);
    std::vector<float> right_upper(num_ranges);

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

        left_lower[i] = static_cast<float>(ll);
        left_upper[i] = static_cast<float>(lr);
        right_lower[i] = static_cast<float>(rl);
        right_upper[i] = static_cast<float>(rr);
    }

    const ScanResult scalar = run_scalar(left_lower, left_upper, right_lower, right_upper,
                                         query_L, query_R);
    const ScanResult simd = run_simd(left_lower, left_upper, right_lower, right_upper,
                                     query_L, query_R);

    std::cout << "SIMD backend: " << simd_backend_name() << '\n';
    std::cout << "Scalar result: " << scalar.last_hit << " took " << scalar.seconds << " seconds\n";
    std::cout << "SIMD result: " << simd.last_hit << " took " << simd.seconds << " seconds\n";
    const double actual_ratio = static_cast<double>(scalar.filtered_indices.size()) / static_cast<double>(num_ranges);
    std::cout << "Candidate ratio: " << actual_ratio * 100.0 << "%\n";
    auto scalar_sorted = scalar.filtered_indices;
    auto simd_sorted = simd.filtered_indices;
    std::sort(scalar_sorted.begin(), scalar_sorted.end());
    std::sort(simd_sorted.begin(), simd_sorted.end());
    if (scalar_sorted != simd_sorted) {
        std::cout << "Warning: scalar and SIMD filtered index sets differ!\n";
    }
    std::cout << '\n';

    return 0;
}
