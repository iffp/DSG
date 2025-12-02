// Author: Zhencan Peng, 2025/11/30
/**
 * @brief Dynamic Segment Graph implementation.
 *
 * The DSG first constructs a temporary HNSW, extracts ef_max-sized neighbor
 * lists for every label, compresses them with a DFS-based dominance filter,
 * and finally serves range queries using the stored forward segment edges.
 */

#include "dsg.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace dsg {

namespace {
using Clock = std::chrono::steady_clock;
using Candidate = std::pair<float, unsigned>;

inline unsigned clampUnsigned(unsigned value, unsigned lo, unsigned hi) {
    return std::max(lo, std::min(value, hi));
}
} // namespace

DynamicSegmentGraph::DynamicSegmentGraph(hnswlib::SpaceInterface<DistType> *space,
                                         const DataWrapper *data) :
    BaseIndex(data),
    space_(space) {
    if (space_ == nullptr) {
        throw std::invalid_argument("DynamicSegmentGraph requires a valid space interface.");
    }
    if (data_wrapper == nullptr) {
        throw std::invalid_argument("DynamicSegmentGraph requires a valid DataWrapper.");
    }
    dist_func_ = space_->get_dist_func();
    dist_func_param_ = space_->get_dist_func_param();
    if (dist_func_ == nullptr) {
        throw std::runtime_error("DynamicSegmentGraph failed to acquire distance function.");
    }
    returned_nns.resize(query_topK);
}

DynamicSegmentGraph::~DynamicSegmentGraph() = default;

void DynamicSegmentGraph::build() {
    if (data_wrapper == nullptr || space_ == nullptr) {
        throw std::runtime_error("DynamicSegmentGraph::build missing prerequisites.");
    }

    const auto build_start = Clock::now();

    forward_edges_.assign(static_cast<std::size_t>(data_wrapper->data_size), {});

    // build temporary HNSW and time it
    auto hnsw_build_start = Clock::now();
    initializeTemporaryHnsw(ef_max);
    for (int label = 0; label < data_wrapper->data_size; ++label) {
        temp_hnsw_->addPoint(static_cast<const void *>(data_wrapper->nodes.at(label)),
                             static_cast<hnswlib::labeltype>(label));
    }
    temp_hnsw_->setEf(ef_max);
    auto hnsw_build_end = Clock::now();
    double hnsw_build_time = std::chrono::duration<double>(hnsw_build_end - hnsw_build_start).count();
    std::cout << "[DSG] Temporary HNSW built in " << hnsw_build_time << " seconds." << std::endl;

    // run KNN for each label
    std::vector<std::pair<unsigned, DistType>> candidates;
    candidates.reserve(ef_max);

    for (int label = 0; label < data_wrapper->data_size; ++label) {
        runKnnForLabel(static_cast<unsigned>(label), ef_max, candidates);
        applyDfsCompression(static_cast<unsigned>(label), candidates);
        storeForwardEdges(static_cast<unsigned>(label));
    }

    temp_hnsw_.reset();

    nodes_amount = 0;
    for (const auto &edges : forward_edges_) {
        nodes_amount += edges.size();
    }
    avg_forward_nns = forward_edges_.empty()
                          ? 0.0F
                          : static_cast<float>(nodes_amount) /
                                static_cast<float>(forward_edges_.size());
    avg_reverse_nns = 0.0F;

    const auto build_end = Clock::now();
    index_time = std::chrono::duration<double>(build_end - build_start).count();
}

void DynamicSegmentGraph::rangeSearch(const float *query,
                                      const std::pair<int, int> query_bound) {
    if (data_wrapper == nullptr || query == nullptr) {
        throw std::invalid_argument("DynamicSegmentGraph::rangeSearch received null inputs.");
    }
    if (forward_edges_.empty()) {
        returned_nns.assign(query_topK, std::numeric_limits<unsigned>::max());
        return;
    }
    if (data_wrapper->data_size <= 0) {
        returned_nns.assign(query_topK, std::numeric_limits<unsigned>::max());
        return;
    }

    const int left = std::max(query_bound.first, 0);
    const int right = std::min(query_bound.second, data_wrapper->data_size - 1);
    if (left > right) {
        returned_nns.assign(query_topK, std::numeric_limits<unsigned>::max());
        return;
    }

    const unsigned left_u = static_cast<unsigned>(left);
    const unsigned right_u = static_cast<unsigned>(right);
    const std::size_t data_size = static_cast<std::size_t>(data_wrapper->data_size);

    auto cmp = [](const Candidate &lhs, const Candidate &rhs) {
        return lhs.first > rhs.first;
    };
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(cmp)> candidate_set(cmp);
    std::priority_queue<Candidate> top_candidates;
    std::vector<std::uint8_t> visited(data_size, 0);

    auto enqueue_seed = [&](unsigned label) {
        if (label < left_u || label > right_u) {
            return;
        }
        if (visited[label]) {
            return;
        }
        visited[label] = 1;
        const DistType dist =
            dist_func_(query, data_wrapper->nodes.at(label), dist_func_param_);
        candidate_set.emplace(dist, label);
        if (top_candidates.size() < query_topK) {
            top_candidates.emplace(dist, label);
        } else if (!top_candidates.empty() && dist < top_candidates.top().first) {
            top_candidates.pop();
            top_candidates.emplace(dist, label);
        }
    };

    const unsigned range_span = right_u - left_u + 1;
    enqueue_seed(left_u);
    enqueue_seed(right_u);
    enqueue_seed(left_u + range_span / 2);
    enqueue_seed(left_u + range_span / 4);

    DistType lower_bound =
        top_candidates.empty() ? std::numeric_limits<DistType>::max()
                               : top_candidates.top().first;

    while (!candidate_set.empty()) {
        const auto [dist, current] = candidate_set.top();
        candidate_set.pop();

        if (top_candidates.size() >= query_topK && dist > lower_bound) {
            break;
        }

        const auto &edges = forward_edges_.at(current);
        for (const auto &edge : edges) {
            if (!edge.coversRange(left_u, right_u)) {
                continue;
            }
            const unsigned neighbor = edge.external_id;
            if (neighbor < left_u || neighbor > right_u) {
                continue;
            }
            if (visited[neighbor]) {
                continue;
            }
            visited[neighbor] = 1;
            const DistType nbr_dist =
                dist_func_(query, data_wrapper->nodes.at(neighbor), dist_func_param_);
            candidate_set.emplace(nbr_dist, neighbor);
            if (top_candidates.size() < query_topK) {
                top_candidates.emplace(nbr_dist, neighbor);
            } else if (nbr_dist < top_candidates.top().first) {
                top_candidates.pop();
                top_candidates.emplace(nbr_dist, neighbor);
            }
            lower_bound =
                top_candidates.empty() ? std::numeric_limits<DistType>::max()
                                       : top_candidates.top().first;
        }
    }

    std::vector<unsigned> ordered_results;
    ordered_results.reserve(top_candidates.size());
    while (!top_candidates.empty()) {
        ordered_results.push_back(top_candidates.top().second);
        top_candidates.pop();
    }
    std::reverse(ordered_results.begin(), ordered_results.end());

    returned_nns.assign(query_topK, std::numeric_limits<unsigned>::max());
    const std::size_t copy_count =
        std::min<std::size_t>(query_topK, ordered_results.size());
    for (std::size_t i = 0; i < copy_count; ++i) {
        returned_nns[i] = ordered_results[i];
    }
}

void DynamicSegmentGraph::save(const std::string &file_path) {
    std::ofstream out(file_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("DynamicSegmentGraph::save failed to open file: " + file_path);
    }
    const std::size_t node_count = forward_edges_.size();
    out.write(reinterpret_cast<const char *>(&node_count), sizeof(node_count));
    for (const auto &edges : forward_edges_) {
        const std::size_t degree = edges.size();
        out.write(reinterpret_cast<const char *>(&degree), sizeof(degree));
        if (degree > 0) {
            out.write(reinterpret_cast<const char *>(edges.data()),
                      static_cast<std::streamsize>(degree * sizeof(SegmentEdge<DistType>)));
        }
    }
}

void DynamicSegmentGraph::load(const std::string &file_path) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("DynamicSegmentGraph::load failed to open file: " + file_path);
    }
    std::size_t node_count = 0;
    in.read(reinterpret_cast<char *>(&node_count), sizeof(node_count));
    if (data_wrapper == nullptr || node_count != static_cast<std::size_t>(data_wrapper->data_size)) {
        throw std::runtime_error("DynamicSegmentGraph::load mismatched dataset size.");
    }

    forward_edges_.assign(node_count, {});
    for (std::size_t node = 0; node < node_count; ++node) {
        std::size_t degree = 0;
        in.read(reinterpret_cast<char *>(&degree), sizeof(degree));
        auto &edges = forward_edges_[node];
        edges.resize(degree);
        if (degree > 0) {
            in.read(reinterpret_cast<char *>(edges.data()),
                    static_cast<std::streamsize>(degree * sizeof(SegmentEdge<DistType>)));
        }
    }

    nodes_amount = 0;
    for (const auto &edges : forward_edges_) {
        nodes_amount += edges.size();
    }
    avg_forward_nns = forward_edges_.empty()
                          ? 0.0F
                          : static_cast<float>(nodes_amount) /
                                static_cast<float>(forward_edges_.size());
    avg_reverse_nns = 0.0F;
}

void DynamicSegmentGraph::initializeTemporaryHnsw(size_t ef_limit) {
    if (space_ == nullptr) {
        throw std::runtime_error("DynamicSegmentGraph::initializeTemporaryHnsw missing space.");
    }
    temp_hnsw_ = std::make_unique<HnswType>(space_, data_wrapper->data_size, M, ef_construction, random_seed);
    temp_hnsw_->ef_max_ = std::max<std::size_t>(ef_limit, static_cast<std::size_t>(M));
}

void DynamicSegmentGraph::runKnnForLabel(
    unsigned label,
    size_t ef_limit,
    std::vector<std::pair<unsigned, DistType>> &candidates) {
    if (!temp_hnsw_) {
        throw std::runtime_error("DynamicSegmentGraph::runKnnForLabel missing temporary HNSW.");
    }
    candidates.clear();
    const void *query = static_cast<const void *>(data_wrapper->nodes.at(label));
    auto queue = temp_hnsw_->searchKnnInternal(const_cast<void *>(query),
                                               static_cast<int>(ef_limit));
    while (!queue.empty()) {
        const auto &entry = queue.top();
        const unsigned neighbor_label =
            static_cast<unsigned>(temp_hnsw_->getExternalLabel(entry.second));
        if (neighbor_label != label) {
            candidates.emplace_back(neighbor_label, entry.first);
        }
        queue.pop();
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.second < rhs.second; });
}

void DynamicSegmentGraph::applyDfsCompression(
    unsigned center_label,
    std::vector<std::pair<unsigned, DistType>> &candidates) {
    const std::size_t max_neighbors = static_cast<std::size_t>(M);
    if (max_neighbors == 0 || candidates.empty()) {
        forward_edges_.at(center_label).clear();
        return;
    }

    auto &ordered = dfs_scratch_.ordered_candidates;
    ordered.assign(candidates.begin(), candidates.end());

    const std::size_t candidate_count = ordered.size();
    auto &prefix = dfs_scratch_.prefix;
    prefix.clear();
    prefix.reserve(max_neighbors);

    dfs_scratch_.is_neighbor.assign(candidate_count, false);
    dfs_scratch_.left_lower.assign(candidate_count, 0);
    dfs_scratch_.left_upper.assign(candidate_count, 0);
    dfs_scratch_.right_lower.assign(candidate_count, 0);
    dfs_scratch_.right_upper.assign(candidate_count, 0);
    dfs_scratch_.domination_cache.clear();

    if (data_wrapper->data_size <= 0) {
        forward_edges_.at(center_label).clear();
        return;
    }

    const unsigned global_left = 0;
    const unsigned global_right =
        static_cast<unsigned>(data_wrapper->data_size - 1);

    auto dfs = [&](auto &&self, unsigned L, unsigned R, unsigned left_cap,
                   unsigned right_cap) -> void {
        if (prefix.size() >= max_neighbors) {
            return;
        }
        const unsigned start_idx = prefix.empty() ? 0 : prefix.back() + 1;
        for (unsigned idx = start_idx; idx < candidate_count; ++idx) {
            const unsigned candidate_label = ordered[idx].first;
            const DistType candidate_dist = ordered[idx].second;
            if (candidate_label < L || candidate_label > R) {
                continue;
            }

            bool dominated = false;
            for (unsigned prev_idx : prefix) {
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(prev_idx) << 32) |
                    static_cast<std::uint64_t>(idx);
                auto cache_it = dfs_scratch_.domination_cache.find(key);
                bool cached = false;
                if (cache_it != dfs_scratch_.domination_cache.end()) {
                    dominated = cache_it->second;
                    cached = true;
                }
                if (!cached) {
                    const DistType pair_dist = dist_func_(
                        data_wrapper->nodes.at(candidate_label),
                        data_wrapper->nodes.at(ordered[prev_idx].first),
                        dist_func_param_);
                    dominated = alpha * pair_dist < candidate_dist;
                    dfs_scratch_.domination_cache.emplace(key, dominated);
                }
                if (dominated) {
                    break;
                }
            }
            if (dominated) {
                continue;
            }

            unsigned next_left_cap = left_cap;
            unsigned next_right_cap = right_cap;
            if (candidate_label < center_label) {
                next_left_cap = std::min(candidate_label, left_cap);
            } else if (candidate_label > center_label) {
                next_right_cap = std::max(candidate_label, right_cap);
            }

            prefix.push_back(idx);
            if (!dfs_scratch_.is_neighbor[idx]) {
                dfs_scratch_.is_neighbor[idx] = true;
                dfs_scratch_.left_lower[idx] = L;
                dfs_scratch_.left_upper[idx] = next_left_cap;
                dfs_scratch_.right_lower[idx] = next_right_cap;
                dfs_scratch_.right_upper[idx] = R;
            } else {
                dfs_scratch_.left_lower[idx] =
                    std::min(dfs_scratch_.left_lower[idx], L);
                dfs_scratch_.left_upper[idx] =
                    std::max(dfs_scratch_.left_upper[idx], next_left_cap);
                dfs_scratch_.right_lower[idx] =
                    std::min(dfs_scratch_.right_lower[idx], next_right_cap);
                dfs_scratch_.right_upper[idx] =
                    std::max(dfs_scratch_.right_upper[idx], R);
            }

            self(self, L, R, next_left_cap, next_right_cap);

            prefix.pop_back();

            if (candidate_label < left_cap) {
                L = candidate_label + 1;
            } else if (candidate_label > right_cap) {
                if (candidate_label == 0) {
                    continue;
                }
                R = candidate_label - 1;
            } else {
                break;
            }

            if (prefix.size() >= max_neighbors) {
                break;
            }
        }
    };

    dfs(dfs, global_left, global_right, center_label, center_label);
}

void DynamicSegmentGraph::storeForwardEdges(unsigned center_label) {
    auto &edges = forward_edges_.at(center_label);
    edges.clear();

    const auto dataset_right =
        data_wrapper->data_size > 0
            ? static_cast<unsigned>(data_wrapper->data_size - 1)
            : 0;

    const std::size_t candidate_count = dfs_scratch_.ordered_candidates.size();
    for (std::size_t idx = 0; idx < candidate_count; ++idx) {
        if (!dfs_scratch_.is_neighbor[idx]) {
            continue;
        }
        const unsigned neighbor_label = dfs_scratch_.ordered_candidates[idx].first;
        if (neighbor_label == center_label) {
            continue;
        }

        const unsigned ll = clampUnsigned(
            std::min(dfs_scratch_.left_lower[idx], dfs_scratch_.left_upper[idx]),
            0U, dataset_right);
        const unsigned lu = clampUnsigned(
            std::max(dfs_scratch_.left_lower[idx], dfs_scratch_.left_upper[idx]),
            0U, dataset_right);
        const unsigned rl = clampUnsigned(
            std::min(dfs_scratch_.right_lower[idx], dfs_scratch_.right_upper[idx]),
            0U, dataset_right);
        const unsigned ru = clampUnsigned(
            std::max(dfs_scratch_.right_lower[idx], dfs_scratch_.right_upper[idx]),
            0U, dataset_right);

        if (ll > lu || rl > ru) {
            continue;
        }

        edges.emplace_back(neighbor_label, ll, lu, rl, ru);
        if (edges.size() >= static_cast<std::size_t>(M)) {
            break;
        }
    }

    std::sort(edges.begin(), edges.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.external_id < rhs.external_id;
              });
}

} // namespace dsg

