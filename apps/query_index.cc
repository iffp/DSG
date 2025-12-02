/**
 * @file query_index.cc
 * @brief Evaluate the Dynamic Segment Graph index on range-filter queries.
 *
 * Usage:
 *   query_index -dataset_path <vectors.bin> -query_path <queries.bin>
 *               -index_path <index.dsg> -groundtruth_root <dir>
 *               [-dataset deep] [-N 100000] [-query_num 1000]
 *               [-query_k 10] [-search_ef 200]
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "base_hnsw/space_l2.h"
#include "data_wrapper.h"
#include "dsg.h"
#include "utils/utils.h"

using namespace std::chrono;

namespace {
struct QueryConfig {
    std::string dataset = "deep";
    int data_size = 100000;
    std::string dataset_path;
    std::string query_path;
    std::string index_path;
    std::string groundtruth_root;
    int query_num = 1000;
    int query_k = 10;
    unsigned search_ef = 200;
};

QueryConfig parseArgs(int argc, char **argv) {
    QueryConfig cfg;
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
        } else if (arg == "-index_path") {
            cfg.index_path = require_value("-index_path");
        } else if (arg == "-groundtruth_root") {
            cfg.groundtruth_root = require_value("-groundtruth_root");
        } else if (arg == "-query_num") {
            cfg.query_num = std::stoi(require_value("-query_num"));
        } else if (arg == "-query_k") {
            cfg.query_k = std::stoi(require_value("-query_k"));
        } else if (arg == "-search_ef") {
            cfg.search_ef = static_cast<unsigned>(std::stoul(require_value("-search_ef")));
        } else if (arg == "-h" || arg == "--help") {
            throw std::invalid_argument("usage");
        }
    }
    if (cfg.dataset_path.empty() || cfg.query_path.empty() || cfg.index_path.empty()) {
        throw std::invalid_argument("dataset_path, query_path, and index_path are required.");
    }
    if (cfg.groundtruth_root.empty()) {
        throw std::invalid_argument("groundtruth_root is required.");
    }
    return cfg;
}

void printUsage() {
    std::cout
        << "Usage: query_index -dataset_path <vectors.bin> -query_path <queries.bin> "
           "-index_path <index.dsg> -groundtruth_root <dir> [options]\n"
        << "Options:\n"
        << "  -dataset <name>          Dataset label (default deep)\n"
        << "  -N <size>                Number of data points (default 100000)\n"
        << "  -query_num <num>         Query count (default 1000)\n"
        << "  -query_k <k>             Query top-K (default 10)\n"
        << "  -search_ef <ef>          Search ef (default 200)\n";
}

std::vector<int> toVector(const std::vector<unsigned> &input, size_t limit) {
    std::vector<int> result;
    result.reserve(limit);
    for (unsigned id : input) {
        if (id == std::numeric_limits<unsigned>::max()) {
            continue;
        }
        result.emplace_back(static_cast<int>(id));
        if (result.size() == limit) {
            break;
        }
    }
    return result;
}
} // namespace

int main(int argc, char **argv) {
    try {
        const QueryConfig cfg = parseArgs(argc, argv);

        DataWrapper data_wrapper(cfg.query_num, cfg.query_k, cfg.dataset, cfg.data_size);
        std::string dataset_path = cfg.dataset_path;
        std::string query_path = cfg.query_path;
        data_wrapper.readData(dataset_path, query_path);
        data_wrapper.LoadGroundtruth(cfg.groundtruth_root);

        base_hnsw::L2Space space(data_wrapper.data_dim);
        dsg::DynamicSegmentGraph index(&space, &data_wrapper);
        index.setQueryTopK(static_cast<unsigned>(cfg.query_k));
        index.setSearchEf(cfg.search_ef);
        index.load(cfg.index_path);

        std::cout << "[DSG] Loaded index from " << cfg.index_path << std::endl;
        std::cout << "[DSG] Running queries with search_ef=" << cfg.search_ef << std::endl;

        for (size_t range_id = 0; range_id < data_wrapper.range_count(); ++range_id) {
            const auto &range_bounds = data_wrapper.query_bounds_by_range(range_id);
            const auto &range_truth = data_wrapper.groundtruth_by_range(range_id);

            if (range_bounds.empty()) {
                continue;
            }

            double precision_acc = 0.0;
            double latency_acc = 0.0;
            size_t evaluated = 0;

            for (size_t query_idx = 0; query_idx < range_bounds.size(); ++query_idx) {
                const auto &bounds = range_bounds[query_idx];
                const float *query_vec = data_wrapper.querys.at(query_idx);

                const auto t0 = steady_clock::now();
                index.rangeSearch(query_vec, bounds);
                const auto t1 = steady_clock::now();

                const auto elapsed = duration<double>(t1 - t0).count();
                latency_acc += elapsed;

                const int *truth_row = range_truth[query_idx];
                auto preds = toVector(index.returned_nns, static_cast<size_t>(cfg.query_k));
                precision_acc += countPrecision(truth_row, static_cast<size_t>(cfg.query_k), preds);
                ++evaluated;
            }

            if (evaluated == 0) {
                continue;
            }

            const double avg_precision = precision_acc / static_cast<double>(evaluated);
            const double avg_latency = latency_acc / static_cast<double>(evaluated);
            const double qps = latency_acc > 0 ? static_cast<double>(evaluated) / latency_acc : 0.0;

            std::cout << std::fixed << std::setprecision(4);
            std::cout << "[Range ratio " << DataWrapper::kRangeRatios[range_id] * 100 << "%] "
                      << "Precision=" << avg_precision << " "
                      << "Latency=" << avg_latency * 1e3 << " ms "
                      << "QPS=" << qps << std::endl;
        }
    } catch (const std::invalid_argument &ex) {
        printUsage();
        std::cerr << "Argument error: " << ex.what() << std::endl;
        return 1;
    } catch (const std::exception &ex) {
        std::cerr << "Query failed: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}