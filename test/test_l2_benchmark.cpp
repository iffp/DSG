/**
 * L2 Distance Benchmark
 * ----------------------
 * Measures the runtime of invoking the `L2Sqr` routine `num_iterations` times
 * on random 96-dimensional vectors. Each iteration performs O(vector_size)
 * arithmetic, so the full benchmark is O(num_iterations * vector_size) time and
 * O(num_iterations * vector_size) space for storing all vectors used.
 */

#include <chrono>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

static float L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const float *lhs = static_cast<const float *>(pVect1v);
    const float *rhs = static_cast<const float *>(pVect2v);
    const size_t qty = *static_cast<const size_t *>(qty_ptr);

    float res = 0.0f;
    for (size_t i = 0; i < qty; ++i) {
        const float diff = lhs[i] - rhs[i];
        res += diff * diff;
    }
    return res;
}

int main() {
    constexpr size_t vector_size = 96;
    constexpr size_t num_iterations = 100;

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<std::pair<std::vector<float>, std::vector<float>>> vector_pairs;
    vector_pairs.reserve(num_iterations);
    for (size_t i = 0; i < num_iterations; ++i) {
        std::vector<float> left(vector_size), right(vector_size);
        for (size_t j = 0; j < vector_size; ++j) {
            left[j] = dist(rng);
            right[j] = dist(rng);
        }
        vector_pairs.emplace_back(std::move(left), std::move(right));
    }

    const size_t qty = vector_size;
    auto start = std::chrono::high_resolution_clock::now();
    float max_dis = 0.0f;
    for (size_t i = 0; i < num_iterations; ++i) {
        const float dis = L2Sqr(vector_pairs[0].first.data(), vector_pairs[i].second.data(), &qty);
        max_dis = std::max(max_dis, dis);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "L2Sqr " << num_iterations << " times with different vectors took: "
              << elapsed.count() << " seconds\n";
    std::cout << max_dis << " max distance\n";

    return 0;
}
