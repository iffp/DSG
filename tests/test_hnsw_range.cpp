// Author: Zhencan Peng, 2025/11/30
/**
 * @file test_hnsw_range.cpp
 * @brief Pure HNSW benchmark for range-filtered queries.
 * @details Loads the Deep-100K dataset through DataWrapper, builds a vanilla
 *          hnswlib::HierarchicalNSW index, and evaluates groundtruth-provided
 *          range bounds via BaseFilterFunctor. Construction requires
 *          O(N log N) time / O(N * M) space while each filtered search costs
 *          O(ef log M) in expectation.
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "base_hnsw/hnswlib.h"
#include "data_wrapper.h"
#include "utils/utils.h"

namespace {

// Filters candidate labels so only nodes within [left_, right_] are accepted.
class RangeFilter : public hnswlib::BaseFilterFunctor {
public:
    RangeFilter(hnswlib::labeltype left, hnswlib::labeltype right)
        : left_(left), right_(right) {}

    bool operator()(hnswlib::labeltype id) override {
        ++evaluations_;
        return id >= left_ && id <= right_;
    }

    [[nodiscard]] std::size_t evaluations() const noexcept { return evaluations_; }

private:
    hnswlib::labeltype left_;
    hnswlib::labeltype right_;
    std::size_t evaluations_ = 0;
};

// Command-line options shared with query_index for apples-to-apples comparison.
struct HnswRangeConfig {
    std::string dataset = "deep";
    int data_size = 100000;
    std::string dataset_path;
    std::string query_path;
    std::string groundtruth_root;
    int query_num = 1000;
    int query_k = 10;
    std::size_t M = 16;
    std::size_t ef_construction = 400;
    std::size_t ef_search = 256;
    unsigned random_seed = 123;
};

// Parse CLI flags; throws std::invalid_argument on malformed input.
HnswRangeConfig parseArgs(int argc, char **argv) {
    HnswRangeConfig cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char *flag) -> const char * {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("Missing value for ") + flag);
            }
            return argv[++i];
        };
        if (arg == "-dataset") {
            cfg.dataset = require_value("-dataset");
        } else if (arg == "-N") {
            cfg.data_size = std::stoi(require_value("-N"));
        } else if (arg == "-dataset_path") {
            cfg.dataset_path = require_value("-dataset_path");
        } else if (arg == "-query_path") {
            cfg.query_path = require_value("-query_path");
        } else if (arg == "-groundtruth_root") {
            cfg.groundtruth_root = require_value("-groundtruth_root");
        } else if (arg == "-query_num") {
            cfg.query_num = std::stoi(require_value("-query_num"));
        } else if (arg == "-query_k") {
            cfg.query_k = std::stoi(require_value("-query_k"));
        } else if (arg == "-M") {
            cfg.M = static_cast<std::size_t>(std::stoul(require_value("-M")));
        } else if (arg == "-ef_construction") {
            cfg.ef_construction = static_cast<std::size_t>(std::stoul(require_value("-ef_construction")));
        } else if (arg == "-ef_search") {
            cfg.ef_search = static_cast<std::size_t>(std::stoul(require_value("-ef_search")));
        } else if (arg == "-seed") {
            cfg.random_seed = static_cast<unsigned>(std::stoul(require_value("-seed")));
        } else if (arg == "-h" || arg == "--help") {
            throw std::invalid_argument("usage");
        }
    }
    if (cfg.dataset_path.empty() || cfg.query_path.empty() || cfg.groundtruth_root.empty()) {
        throw std::invalid_argument(
            "dataset_path, query_path, and groundtruth_root are required for pure HNSW benchmarking.");
    }
    return cfg;
}

void printUsage() {
    std::cout
        << "Usage: test_hnsw_range -dataset_path <vectors.bin> -query_path <queries.bin> "
           "-groundtruth_root <dir> [options]\n"
        << "Options:\n"
        << "  -dataset <name>          Dataset tag (default deep)\n"
        << "  -N <size>                Number of base vectors (default 100000)\n"
        << "  -query_num <num>         Query count (default 1000)\n"
        << "  -query_k <k>             Result top-K (default 10)\n"
        << "  -M <degree>              HNSW out-degree (default 16)\n"
        << "  -ef_construction <ef>    Build ef (default 400)\n"
        << "  -ef_search <ef>          Search ef (default 256)\n"
        << "  -seed <value>            Random seed (default 123)\n";
}

// Convert knn results (distance,label) into a compact top-K vector for scoring.
std::vector<int> toVector(const std::vector<std::pair<float, hnswlib::labeltype>> &input,
                          std::size_t limit) {
    std::vector<int> result;
    result.reserve(limit);
    for (const auto &entry : input) {
        result.emplace_back(static_cast<int>(entry.second));
        if (result.size() == limit) {
            break;
        }
    }
    return result;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const HnswRangeConfig cfg = parseArgs(argc, argv);

        // Reuse the existing binary readers so we exactly match DSG datasets.
        std::string dataset_path = cfg.dataset_path;
        std::string query_path = cfg.query_path;
        std::string gt_root = cfg.groundtruth_root;

        DataWrapper data_wrapper(cfg.query_num, cfg.query_k, cfg.dataset, cfg.data_size);
        data_wrapper.readData(dataset_path, query_path);
        data_wrapper.LoadGroundtruth(gt_root);

        // Build a plain HNSW index that mirrors the DSG hyper-parameters.
        std::cout << "[PureHNSW] dataset=" << cfg.dataset << " N=" << cfg.data_size
                  << " queries=" << cfg.query_num << " topK=" << cfg.query_k
                  << " M=" << cfg.M << " ef_build=" << cfg.ef_construction
                  << " ef_search=" << cfg.ef_search << std::endl;

        hnswlib::L2Space space(data_wrapper.data_dim);
        hnswlib::HierarchicalNSW<float> index(
            &space, static_cast<std::size_t>(cfg.data_size),
            cfg.M, cfg.ef_construction, cfg.random_seed);

        for (int label = 0; label < cfg.data_size; ++label) {
            index.addPoint(static_cast<const void *>(data_wrapper.nodes.at(label)),
                           static_cast<hnswlib::labeltype>(label));
        }
        index.setEf(cfg.ef_search);

        using clock = std::chrono::steady_clock;
        // Iterate over every range ratio block to accumulate recall/latency stats.
        for (size_t range_id = 0; range_id < data_wrapper.range_count(); ++range_id) {
            const auto &range_bounds = data_wrapper.query_bounds_by_range(range_id);
            const auto &range_truth = data_wrapper.groundtruth_by_range(range_id);
            if (range_bounds.empty()) {
                continue;
            }

            double recall_acc = 0.0;
            double latency_acc = 0.0;
            std::size_t evaluated = 0;

            for (size_t query_idx = 0; query_idx < range_bounds.size(); ++query_idx) {
                const auto &bounds = range_bounds[query_idx];
                const float *query_vec = data_wrapper.querys.at(query_idx);

                RangeFilter filter(static_cast<hnswlib::labeltype>(bounds.first),
                                   static_cast<hnswlib::labeltype>(bounds.second));

                const auto t0 = clock::now();
                const auto results = index.searchKnnCloserFirst(
                    query_vec, static_cast<std::size_t>(cfg.query_k), &filter);
                const auto t1 = clock::now();

                latency_acc += std::chrono::duration<double>(t1 - t0).count();

                const int *truth_row = range_truth[query_idx];
                auto preds = toVector(results, static_cast<std::size_t>(cfg.query_k));
                recall_acc += countRecall(
                    truth_row, static_cast<std::size_t>(cfg.query_k), preds);
                ++evaluated;
            }

            if (evaluated == 0) {
                continue;
            }

            const double avg_recall = recall_acc / static_cast<double>(evaluated);
            const double avg_latency = latency_acc / static_cast<double>(evaluated);
            const double qps = latency_acc > 0 ? static_cast<double>(evaluated) / latency_acc : 0.0;

            std::cout << std::fixed << std::setprecision(4);
            std::cout << "[PureHNSW range ratio "
                      << DataWrapper::kRangeRatios[range_id] * 100 << "%] "
                      << "Recall=" << avg_recall << " "
                      << "Latency=" << avg_latency * 1e3 << " ms "
                      << "QPS=" << qps << std::endl;
        }
    } catch (const std::invalid_argument &ex) {
        printUsage();
        std::cerr << "Argument error: " << ex.what() << std::endl;
        return 1;
    } catch (const std::exception &ex) {
        std::cerr << "Pure HNSW benchmark failed: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}

