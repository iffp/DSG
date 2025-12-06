// Author: Zhencan Peng, 2025/11/30
/**
 * @brief Dynamic Segment Graph implementation.
 *
 * The DSG first constructs a temporary HNSW, extracts ef_max-sized neighbor
 * lists for every label, compresses them with a DFS-based dominance filter,
 * and finally serves range queries using the stored forward segment edges.
 * This implementation is based on the paper "Dynamic Segment Graph for Approximate Nearest Neighbor Search" by Zhencan Peng et al.
 * And previously we use compact_graph.h for arbitrary query and insertion. But now we restructure the code to be more modular and easy to understand.
 */

#include "dsg.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <unordered_set>
#include <stdexcept>
#include <iostream>

namespace dsg {

namespace {
using Clock = std::chrono::steady_clock;
using Candidate = std::pair<DynamicSegmentGraph::DistType, unsigned>;
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
    visited_list_pool_ = new hnswlib::VisitedListPool(1, data_wrapper->data_size);
    returned_nns.resize(query_topK);
}

DynamicSegmentGraph::~DynamicSegmentGraph() {
    temp_hnsw_.reset();
    if (visited_list_pool_) {
        delete visited_list_pool_;
        visited_list_pool_ = nullptr;
    }
}

void DynamicSegmentGraph::build() {

    const auto build_start = Clock::now();

    forward_edges_.assign(static_cast<std::size_t>(data_wrapper->data_size), {});
    const std::size_t node_count = static_cast<std::size_t>(data_wrapper->data_size);
    std::vector<std::vector<std::pair<unsigned, DistType>>> all_candidates(node_count);
    
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
    double total_knn_time = 0.0;
    double total_merge_time = 0.0;
    double total_dfs_time = 0.0;
    double total_store_time = 0.0;

    for (int label = 0; label < data_wrapper->data_size; ++label) {
        auto t1 = Clock::now();
        runKnnForLabel(static_cast<unsigned>(label), ef_max, all_candidates.at(static_cast<std::size_t>(label)));
        auto t2 = Clock::now();
        total_knn_time += std::chrono::duration<double>(t2 - t1).count();
    }

    // add reverse edges so each point sees incoming sources
    for (std::size_t label = 0; label < node_count; ++label) {
        const auto &forward = all_candidates[label];
        for (const auto &entry : forward) {
            const unsigned neighbor = entry.first;
            if (neighbor >= node_count) {
                continue;
            }
            all_candidates[neighbor].emplace_back(static_cast<unsigned>(label), entry.second);
        }
    }

    for (std::size_t label = 0; label < node_count; ++label) {
        auto &candidates = all_candidates[label];

        auto t_merge_start = Clock::now();
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto &lhs, const auto &rhs) {
                      if (lhs.second == rhs.second) {
                          return lhs.first < rhs.first;
                      }
                      return lhs.second < rhs.second;
                  });
        std::unordered_set<unsigned> seen;
        seen.reserve(candidates.size());
        std::vector<std::pair<unsigned, DistType>> deduped;
        deduped.reserve(candidates.size());
        for (const auto &entry : candidates) {
            if (seen.insert(entry.first).second) {
                deduped.push_back(entry);
            }
        }
        candidates.swap(deduped);
        auto t_merge_end = Clock::now();

        auto t_dfs_start = Clock::now();
        applyDfsCompression(static_cast<unsigned>(label), candidates);
        auto t_dfs_end = Clock::now();
        storeForwardEdges(static_cast<unsigned>(label));
        auto t_store_end = Clock::now();

        total_merge_time += std::chrono::duration<double>(t_merge_end - t_merge_start).count();
        total_dfs_time += std::chrono::duration<double>(t_dfs_end - t_dfs_start).count();
        total_store_time += std::chrono::duration<double>(t_store_end - t_dfs_end).count();
    }

    // After all forward edges are stored, append reverse edges and merge duplicates.
    std::vector<std::vector<SegmentEdge<DistType>>> reverse_edges(node_count);
    for (std::size_t src = 0; src < node_count; ++src) {
        for (const auto &edge : forward_edges_[src]) {
            const unsigned dst = edge.external_id;
            if (dst >= node_count) {
                continue;
            }
            reverse_edges[dst].emplace_back(static_cast<unsigned>(src),
                                            edge.left_lower,
                                            edge.left_upper,
                                            edge.right_lower,
                                            edge.right_upper);
        }
    }

    auto merge_edges = [](std::vector<SegmentEdge<DistType>> &edges) {
        if (edges.empty()) {
            return;
        }
        std::sort(edges.begin(), edges.end(),
                  [](const auto &a, const auto &b) {
                      if (a.external_id == b.external_id) {
                          return std::tie(a.left_lower, a.left_upper, a.right_lower, a.right_upper) <
                                 std::tie(b.left_lower, b.left_upper, b.right_lower, b.right_upper);
                      }
                      return a.external_id < b.external_id;
                  });
        std::vector<SegmentEdge<DistType>> merged;
        merged.reserve(edges.size());
        merged.push_back(edges.front());
        for (std::size_t i = 1; i < edges.size(); ++i) {
            auto &last = merged.back();
            const auto &cur = edges[i];
            if (last.external_id == cur.external_id) {
                last.left_lower = std::min(last.left_lower, cur.left_lower);
                last.left_upper = std::max(last.left_upper, cur.left_upper);
                last.right_lower = std::min(last.right_lower, cur.right_lower);
                last.right_upper = std::max(last.right_upper, cur.right_upper);
            } else {
                merged.push_back(cur);
            }
        }
        edges.swap(merged);
    };

    for (std::size_t label = 0; label < node_count; ++label) {
        auto &edges = forward_edges_[label];
        auto &rev = reverse_edges[label];
        edges.insert(edges.end(), rev.begin(), rev.end());
        merge_edges(edges);
    }

    const auto build_end = Clock::now();
    index_time = std::chrono::duration<double>(build_end - build_start).count();
    
    std::cout << "[DSG] Detailed breakdown:" << std::endl;
    std::cout << "  KNN Search: " << total_knn_time << " s" << std::endl;
    std::cout << "  Merge/Dedup: " << total_merge_time << " s" << std::endl;
    std::cout << "  DFS Compress: " << total_dfs_time << " s" << std::endl;
    std::cout << "  Store Edges: " << total_store_time << " s" << std::endl;

}

void DynamicSegmentGraph::rangeSearch(const float *query,
                                      const std::pair<int, int> query_bound) {

    const int left = query_bound.first;
    const int right = query_bound.second;

    const unsigned left_u = static_cast<unsigned>(left);
    const unsigned right_u = static_cast<unsigned>(right);

    hnswlib::VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    hnswlib::vl_type *visited_array = vl->mass;
    hnswlib::vl_type visited_array_tag = vl->curV;
    std::size_t hop_counter = 0;
    std::size_t distance_eval_count = 0;

    auto timed_distance = [&](unsigned label) -> DistType {
        ++distance_eval_count;
        const DistType dist =
            dist_func_(query, data_wrapper->nodes[label], dist_func_param_);
        return dist;
    };

    auto cmp = [](const Candidate &lhs, const Candidate &rhs) {
        return lhs.first > rhs.first;
    };
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(cmp)> candidate_set(cmp);
    std::priority_queue<Candidate> top_candidates;
    std::vector<unsigned> fetched_nns;
    fetched_nns.reserve(search_ef);

    auto enqueue_seed = [&](unsigned label) {
        if (visited_array[label] == visited_array_tag) {
            return;
        }
        visited_array[label] = visited_array_tag;
        const DistType dist = timed_distance(label);
        candidate_set.emplace(dist, label);
        
    };

    const unsigned range_span = right_u - left_u + 1;
    enqueue_seed(left_u);
    enqueue_seed(left_u + range_span / 2);
    enqueue_seed(left_u + range_span / 4);
    enqueue_seed(left_u + 3 * range_span / 4);

    DistType lower_bound = std::numeric_limits<DistType>::max();

    while (!candidate_set.empty()) {
        const auto [dist, current] = candidate_set.top();
        candidate_set.pop();
        ++hop_counter;

        if (dist > lower_bound) {
            break;
        }

        const auto &edges = forward_edges_.at(current);
        fetched_nns.clear();

        // Use binary search to find the first neighbor >= left_u
        auto it = std::lower_bound(edges.begin(), edges.end(), left_u,
                                   [](const SegmentEdge<DistType> &edge, unsigned val) {
                                       return edge.external_id < val;
                                   });
                                   
        for (; it != edges.end(); ++it) {
            const auto &edge = *it;
            const unsigned neighbor = edge.external_id;

            // Since edges are sorted by external_id, we can break early
            if (neighbor > right_u) {
                break;
            }

            if (!edge.coversRange(left_u, right_u)) {
                continue;
            }
            
            if (visited_array[neighbor] == visited_array_tag) {
                continue;
            }
            fetched_nns.push_back(neighbor);
        }

        for (const auto neighbor : fetched_nns) {
            visited_array[neighbor] = visited_array_tag;
            const DistType nbr_dist = timed_distance(neighbor);
            
            if (top_candidates.size() < search_ef) {
                candidate_set.emplace(nbr_dist, neighbor);
                top_candidates.emplace(nbr_dist, neighbor);
                lower_bound = top_candidates.top().first;
            } else if (nbr_dist < lower_bound) {
                candidate_set.emplace(nbr_dist, neighbor);
                top_candidates.emplace(nbr_dist, neighbor);
                top_candidates.pop();
                lower_bound = top_candidates.top().first;
            }
        }
    }

    visited_list_pool_->releaseVisitedList(vl);

    while (top_candidates.size() > query_topK) {
        top_candidates.pop();
    }

    returned_nns.clear();
    while (!top_candidates.empty()) {
        returned_nns.emplace_back(top_candidates.top().second);
        top_candidates.pop();
    }

    last_hop_count_ = hop_counter;
    last_distance_eval_count_ = distance_eval_count;
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

    edges_amount = 0;
    for (const auto &edges : forward_edges_) {
        edges_amount += edges.size();
    }
    avg_forward_nns = forward_edges_.empty()
                          ? 0.0F
                          : static_cast<float>(edges_amount) /
                                static_cast<float>(forward_edges_.size());
    avg_reverse_nns = 0.0F;
}

void DynamicSegmentGraph::getStats() {
    // calculate the average number of forward neighbors
    edges_amount = 0;
    for (const auto &edges : forward_edges_) {
        edges_amount += edges.size();
    }
    avg_forward_nns = forward_edges_.empty()
                          ? 0.0F
                          : static_cast<float>(edges_amount) /
                                static_cast<float>(forward_edges_.size());
    avg_reverse_nns = 0.0F;

    std::cout << "DynamicSegmentGraph Statistics:" << std::endl;
    std::cout << "  Build Time: " << index_time << " seconds" << std::endl;
    std::cout << "  Total Edges: " << edges_amount << std::endl;
    std::cout << "  Avg Forward NNs: " << avg_forward_nns << std::endl;
    std::cout << "  Avg Reverse NNs: " << avg_reverse_nns << std::endl;
}

void DynamicSegmentGraph::initializeTemporaryHnsw(size_t ef_limit) {
    if (space_ == nullptr) {
        throw std::runtime_error("DynamicSegmentGraph::initializeTemporaryHnsw missing space.");
    }
    temp_hnsw_ = std::make_unique<HnswType>(space_, data_wrapper->data_size, M, ef_construction, random_seed);
    // setEF is to set the ef for the search in HNSW
    temp_hnsw_->setEf(ef_limit);
}

void DynamicSegmentGraph::runKnnForLabel(
    unsigned label,
    size_t ef_limit,
    std::vector<std::pair<unsigned, DistType>> &candidates) {
    candidates.clear();
    const void *query = static_cast<const void *>(data_wrapper->nodes[label]);
    // Use searchKnnCloserFirstReturnAll to get ALL candidates that were ever considered
    // during the search, not just the top ef results. This provides more candidates
    // for DFS compression, similar to compact_graph.h's searchBaseLayerLevel0.
    const auto results = temp_hnsw_->searchKnnCloserFirstReturnAll(query, ef_limit);
    for (const auto &entry : results) {
        const unsigned neighbor_label = static_cast<unsigned>(entry.second);
        if (neighbor_label == label) {
            continue;
        }
        candidates.emplace_back(neighbor_label, entry.first);
    }
}

void DynamicSegmentGraph::applyDfsCompression(
    unsigned center_label,
    std::vector<std::pair<unsigned, DistType>> &candidates) {
    const std::size_t max_neighbors = static_cast<std::size_t>(M);

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
                    dfs_scratch_.domination_cache[key] = dominated;
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

    const std::size_t candidate_count = dfs_scratch_.ordered_candidates.size();
    for (std::size_t idx = 0; idx < candidate_count; ++idx) {
        if (!dfs_scratch_.is_neighbor[idx]) {
            continue;
        }
        const unsigned neighbor_label = dfs_scratch_.ordered_candidates[idx].first;
        const unsigned ll = dfs_scratch_.left_lower[idx];
        const unsigned lu = dfs_scratch_.left_upper[idx];
        const unsigned rl = dfs_scratch_.right_lower[idx];
        const unsigned ru = dfs_scratch_.right_upper[idx];

        edges.emplace_back(neighbor_label, ll, lu, rl, ru);
    }
}

} // namespace dsg

