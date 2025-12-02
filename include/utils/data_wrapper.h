/**
 * @file data_vecs.h
 * @brief Manage raw data vectors and query metadata.
 *
 * This class encapsulates dataset metadata, nodes, queries, precomputed ranges,
 * and groundtruth vectors for benchmarking range filters. We assume the dataset
 * vectors are pre-sorted and each vector receives a deterministic attribute id
 * equal to its index in [0, data_size - 1].
 *
 */

#pragma once

#include <array>
#include <string>
#include <vector>

#include "flat_vectors.h"

// 使用标准库中的pair、string和vector类型
using std::pair;
using std::string;
using std::vector;

class DataWrapper {
public:
    static constexpr size_t kStaticRangeCount = 7;
    inline static constexpr std::array<double, kStaticRangeCount> kRangeRatios{
        0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64};
    static constexpr int kStaticTopK = 10;
    static constexpr int kStaticQueryNum = 1000;

    /**
     * Constructor initializes dataset name, size, query count, and top-k target.
     */
    DataWrapper(int num, int k_, string dataset_name, int data_size_)
        : dataset(dataset_name),   // Dataset name (constant).
          data_size(data_size_),   // Dataset size (constant).
          query_num(num),          // Query count (constant).
          query_k(k_) {}           // Target top-k (constant).

    // Dataset name (constant).
    const string dataset;
    
    // Total number of vectors (constant).
    const int data_size;
    
    // Number of queries (constant).
    const int query_num;
    
    // Target top-k (constant).
    const int query_k;
    
    // Dimensionality of the dataset vectors.
    size_t data_dim;
    
    // Stored dataset vectors.
    FlatVectors<float> nodes;
    
    // Raw query vectors.
    FlatVectors<float> querys;
    
    // Range bounds for static queries.
    std::array<vector<pair<int, int>>, kStaticRangeCount> static_query_ranges;
    
    // Groundtruth per static range bucket.
    std::array<FlatVectors<int>, kStaticRangeCount> static_groundtruth;
    
    /**
     * Load dataset and query files from disk.
     */
    void readData(string &dataset_path, string &query_path);
    
    /**
     * Load groundtruth from persistent storage.
     */
    void LoadGroundtruth(const string &gt_root);

    /**
     * Generate range-filtering queries and matching groundtruth (benchmark edition).
     */
    void generateRangeFilteringQueriesAndGroundtruthBenchmark(const string &save_root = "./groundtruth/static");

    void generateIncrementalInsertionGroundtruth(int num_parts, const string &save_dir);

    size_t range_count() const { return kStaticRangeCount; }
    const FlatVectors<int> &groundtruth_by_range(size_t range_id) const {
        return static_groundtruth.at(range_id);
    }
    const vector<pair<int, int>> &query_bounds_by_range(size_t range_id) const {
        return static_query_ranges.at(range_id);
    }
};
