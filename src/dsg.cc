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
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <unordered_set>
#include <stdexcept>
#include <iostream>
#include <tuple>
#include <xmmintrin.h>
#include <immintrin.h>

namespace dsg {

namespace {
using Clock = std::chrono::steady_clock;
using Candidate = std::pair<DynamicSegmentGraph::DistType, unsigned>;

static inline bool EnvVarEnabled(const char *name, bool default_value) noexcept {
    const char *val = std::getenv(name);
    if (val == nullptr) {
        return default_value;
    }
    while (*val == ' ' || *val == '\t' || *val == '\n' || *val == '\r') {
        ++val;
    }
    if (*val == '\0') {
        return default_value;
    }
    // Keep parsing cheap. Prefer numeric toggles: 0/1.
    const char c = *val;
    if (c == '0' || c == 'f' || c == 'F' || c == 'n' || c == 'N') {
        return false;
    }
    if (c == '1' || c == 't' || c == 'T' || c == 'y' || c == 'Y') {
        return true;
    }
    return default_value;
}

// Temporary structure used only during build process for sorting/merging.
struct TempEdge {
    unsigned external_id;
    unsigned left_lower;
    unsigned left_upper;
    unsigned right_lower;
    unsigned right_upper;

    bool coversRange(unsigned query_left, unsigned query_right) const noexcept {
        return (left_lower <= query_left && query_left <= left_upper) &&
               (right_lower <= query_right && query_right <= right_upper);
    }
};

// Support heuristic for an edge (center -> v) under the full space of query ranges.
// A query range [L, R] can use this edge only if:
//   - L in [ll, lu]
//   - R in [rl, ru]
//   - L <= v <= R   (because adjacency is scanned only for neighbor ids in [L, R])
//
// The support is the number of (L, R) pairs satisfying these constraints:
//   support = |{L}| * |{R}|
static inline std::uint64_t EdgeSupport(unsigned v,
                                       unsigned ll,
                                       unsigned lu,
                                       unsigned rl,
                                       unsigned ru) noexcept {
    const std::uint64_t count_L = static_cast<std::uint64_t>(lu - ll + 1);
    const std::uint64_t count_R = static_cast<std::uint64_t>(ru - rl + 1);
    return count_L * count_R;
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

    // Trigger: control whether to add reverse KNN candidates before compression.
    // Default is OFF (enable by exporting: DSG_ADD_REVERSE_KNN_EDGES=1).
    const bool add_reverse_knn_edges =
        EnvVarEnabled("DSG_ADD_REVERSE_KNN_EDGES", false);
    std::cout << "[DSG] Add reverse KNN edges: "
              << (add_reverse_knn_edges ? "ON" : "OFF")
              << " (env: DSG_ADD_REVERSE_KNN_EDGES)"
              << std::endl;

    const std::size_t node_count = static_cast<std::size_t>(data_wrapper->data_size);
    // Use a temporary vector of vectors for building, merging, and sorting edges.
    std::vector<std::vector<TempEdge>> temp_adj(node_count);
    
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
    if (add_reverse_knn_edges) {
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
    }

    // Local lambda to store forward edges into temp_adj from dfs_scratch_
    auto store_edges_locally = [&](unsigned center_label) {
        auto &edges = temp_adj.at(center_label);
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

            edges.push_back(TempEdge{neighbor_label, ll, lu, rl, ru});
        }
    };

    // Local lambda to normalize candidate ordering before DFS compression.
    // - With reverse-KNN augmentation enabled, we may introduce duplicate ids, so we
    //   must sort by id + unique first, then sort by distance (nearest-first).
    // - Without reverse-KNN augmentation, `runKnnForLabel()` already emits unique
    //   candidates in nearest-first order (after heap extraction). No extra work needed.
    auto prepare_candidates_for_dfs = [&](std::vector<std::pair<unsigned, DistType>> &candidates) {
        if (add_reverse_knn_edges) {
            std::sort(candidates.begin(), candidates.end(),
                      [](const auto &lhs, const auto &rhs) {
                          return lhs.first < rhs.first;
                      });

            candidates.erase(std::unique(candidates.begin(), candidates.end(),
                                         [](const auto &lhs, const auto &rhs) {
                                             return lhs.first == rhs.first;
                                         }),
                             candidates.end());

            // Primary: Distance (asc), Secondary: ID (asc).
            std::sort(candidates.begin(), candidates.end(),
                      [](const auto &lhs, const auto &rhs) {
                          if (std::abs(lhs.second - rhs.second) > 1e-6f) {
                              return lhs.second < rhs.second;
                          }
                          return lhs.first < rhs.first;
                      });
            return;
        }
    };

    for (std::size_t label = 0; label < node_count; ++label) {
        auto &candidates = all_candidates[label];

        const auto t_prepare_start = Clock::now();
        prepare_candidates_for_dfs(candidates);
        const auto t_prepare_end = Clock::now();

        const auto t_dfs_start = Clock::now();
        applyDfsCompression(static_cast<unsigned>(label), candidates);
        const auto t_dfs_end = Clock::now();
        store_edges_locally(static_cast<unsigned>(label));
        const auto t_store_end = Clock::now();

        total_merge_time += std::chrono::duration<double>(t_prepare_end - t_prepare_start).count();
        total_dfs_time += std::chrono::duration<double>(t_dfs_end - t_dfs_start).count();
        total_store_time += std::chrono::duration<double>(t_store_end - t_dfs_end).count();
    }

    // After all forward edges are stored, append reverse edges and merge duplicates.
    std::vector<std::vector<TempEdge>> reverse_edges(node_count);
    for (std::size_t src = 0; src < node_count; ++src) {
        for (const auto &edge : temp_adj[src]) {
            const unsigned dst = edge.external_id;
            if (dst >= node_count) {
                continue;
            }
            reverse_edges[dst].push_back(TempEdge{static_cast<unsigned>(src),
                                            edge.left_lower,
                                            edge.left_upper,
                                            edge.right_lower,
                                            edge.right_upper});
        }
    }

    auto merge_edges_func = [](std::vector<TempEdge> &edges) {
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
        std::vector<TempEdge> merged;
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
        auto &edges = temp_adj[label];
        auto &rev = reverse_edges[label];
        edges.insert(edges.end(), rev.begin(), rev.end());
        merge_edges_func(edges);
    }

    // ---------------------------------------------------------------------
    // Build-time support pruning (bottom 10% support per node).
    //
    // Motivation:
    // - Very low-support edges correspond to tiny (L, R) eligibility regions.
    // - These edges rarely contribute across the full query-range space.
    //
    // Small-range guard:
    // - `rangeSearch()` skips envelope checks when range_span < 0.02 * N.
    // - For such tiny ranges, useful edges tend to connect labels close to the
    //   current node (center_label). To avoid harming this regime, we never prune
    //   edges whose neighbor label lies within +/- ceil(0.02 * N) around center_label.
    // ---------------------------------------------------------------------
    const auto prune_start = Clock::now();
    const std::size_t protect_span = (node_count + 49) / 50; // ceil(0.02 * N) = ceil(N / 50)
    // Distance-weighted pruning score to discourage far-in-label edges:
    // score = support / |center_label - neighbor_label|.
    std::cout << "[DSG] Build-time support prune: using distance-weighted score support/|nbr-center|"
              << std::endl;
    std::size_t edges_before_prune = 0;
    std::size_t edges_after_prune = 0;
    std::size_t pruned_edges = 0;
    std::size_t protected_edges = 0;

    std::vector<std::uint64_t> supports_all;
    std::vector<std::uint64_t> prunable_supports;
    std::vector<TempEdge> kept_edges;

    const std::uint64_t kProtectedSentinel =
        std::numeric_limits<std::uint64_t>::max();

    for (std::size_t center_label = 0; center_label < node_count; ++center_label) {
        auto &edges = temp_adj[center_label];
        edges_before_prune += edges.size();
        if (edges.empty()) {
            continue;
        }

        supports_all.resize(edges.size());
        prunable_supports.clear();
        prunable_supports.reserve(edges.size());

        for (std::size_t i = 0; i < edges.size(); ++i) {
            const auto &edge = edges[i];
            const std::size_t neighbor = static_cast<std::size_t>(edge.external_id);
            const std::size_t diff =
                neighbor > center_label ? (neighbor - center_label + 1)
                                        : (center_label - neighbor + 1);
            if (diff <= protect_span) {
                supports_all[i] = kProtectedSentinel;
                ++protected_edges;
                continue;
            }

            std::uint64_t support = EdgeSupport(edge.external_id,
                                                edge.left_lower,
                                                edge.left_upper,
                                                edge.right_lower,
                                                edge.right_upper);
            support /= static_cast<std::uint64_t>(diff);
            supports_all[i] = support;
            prunable_supports.push_back(support);
        }

        const std::size_t prunable_count = prunable_supports.size();
        const std::size_t prune_count = prunable_count / 10;
        if (prune_count == 0) {
            edges_after_prune += edges.size();
            continue;
        }

        const std::size_t kth = prune_count - 1;
        std::nth_element(prunable_supports.begin(),
                         prunable_supports.begin() + kth,
                         prunable_supports.end());
        const std::uint64_t cutoff = prunable_supports[kth];

        kept_edges.clear();
        kept_edges.reserve(edges.size());
        for (std::size_t i = 0; i < edges.size(); ++i) {
            const auto &edge = edges[i];
            const std::uint64_t support = supports_all[i];
            if (support == kProtectedSentinel) {
                kept_edges.push_back(edge);
                continue;
            }
            if (support <= cutoff) {
                ++pruned_edges;
                continue;
            }
            kept_edges.push_back(edge);
        }
        edges.swap(kept_edges);
        edges_after_prune += edges.size();
    }

    // If we skipped pruning for some nodes (e.g. prune_count==0), edges_after_prune
    // already includes their original degree. For nodes that were empty, we added 0.
    // For nodes pruned, edges_after_prune was added after pruning.
    if (edges_after_prune == 0) {
        for (const auto &edges : temp_adj) {
            edges_after_prune += edges.size();
        }
    }

    const auto prune_end = Clock::now();
    const double prune_time_s =
        std::chrono::duration<double>(prune_end - prune_start).count();
    const double prune_frac = edges_before_prune == 0
                                  ? 0.0
                                  : static_cast<double>(pruned_edges) /
                                        static_cast<double>(edges_before_prune);

    std::cout << "[DSG] Build-time support prune (per node, bottom 10% of prunable edges): "
              << "pruned=" << pruned_edges << "/" << edges_before_prune
              << " (" << prune_frac * 100.0 << "%), "
              << "protected_edges=" << protected_edges
              << " (|nbr-center| <= " << protect_span << "), "
              << "score=support/|nbr-center|, "
              << "time=" << prune_time_s << " s" << std::endl;

    // Now flatten temp_adj into SoA members
    row_offset_.resize(node_count + 1);
    std::size_t total_edges = 0;
    for (const auto &edges : temp_adj) {
        total_edges += edges.size();
    }
    neighbors_.resize(total_edges);
    left_lower_.resize(total_edges);
    left_upper_.resize(total_edges);
    right_lower_.resize(total_edges);
    right_upper_.resize(total_edges);

    std::size_t current_offset = 0;
    for (std::size_t i = 0; i < node_count; ++i) {
        row_offset_[i] = current_offset;
        for (const auto &edge : temp_adj[i]) {
            neighbors_[current_offset] = edge.external_id;
            left_lower_[current_offset] = edge.left_lower;
            left_upper_[current_offset] = edge.left_upper;
            right_lower_[current_offset] = edge.right_lower;
            right_upper_[current_offset] = edge.right_upper;
            current_offset++;
        }
    }
    row_offset_[node_count] = current_offset;

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
    constexpr double kSmallRangeFrac = 0.02;
    // Small optimization: if the query span is tiny, skip envelope checks and
    // admit neighbors inside [left_u, right_u] to avoid over-pruning sparse ranges.
    const bool skip_range_envelope =
        static_cast<double>(range_span) < kSmallRangeFrac * static_cast<double>(data_wrapper->data_size);
    enqueue_seed(left_u);
    enqueue_seed(left_u + range_span / 2);
    enqueue_seed(left_u + range_span / 4);
    enqueue_seed(left_u + 3 * range_span / 4);

    DistType lower_bound = std::numeric_limits<DistType>::max();

    // Prepare SIMD constants
    // Note: using signed comparison because standard _mm_cmple_epi32 is signed. 
    // For unsigned comparison in SSE2/AVX2, we toggle the sign bit (0x80000000).
    // (val ^ 0x80000000) > (bound ^ 0x80000000)
    const __m128i sign_bit = _mm_set1_epi32(0x80000000);
    const __m128i v_left_u = _mm_set1_epi32(static_cast<int>(left_u));
    const __m128i v_right_u = _mm_set1_epi32(static_cast<int>(right_u));
    const __m128i v_left_u_adj = _mm_xor_si128(v_left_u, sign_bit);
    const __m128i v_right_u_adj = _mm_xor_si128(v_right_u, sign_bit);

    // Pointers for SoA arrays
    const unsigned* neighbors_ptr = neighbors_.data();
    const unsigned* ll_ptr = left_lower_.data();
    const unsigned* lu_ptr = left_upper_.data();
    const unsigned* rl_ptr = right_lower_.data();
    const unsigned* ru_ptr = right_upper_.data();

    while (!candidate_set.empty()) {
        const auto [dist, current] = candidate_set.top();
        candidate_set.pop();
        ++hop_counter;

        if (dist > lower_bound) {
            break;
        }

        // SoA access
        const size_t start_idx = row_offset_[current];
        const size_t end_idx = row_offset_[current + 1];

        fetched_nns.clear();

        // Binary search in neighbors_ array for the range
        auto start_it = neighbors_.begin() + start_idx;
        auto end_it = neighbors_.begin() + end_idx;

        auto it = std::lower_bound(start_it, end_it, left_u);
        
        // Calculate new start index based on lower_bound result
        size_t current_scan_idx = std::distance(neighbors_.begin(), it);

        // SIMD Loop: process 4 elements at a time
        // Adjust loop start to current_scan_idx instead of start_idx
        if (!skip_range_envelope) {
            for (; current_scan_idx + 4 <= end_idx; current_scan_idx += 4) {
                // Prefetch ahead (16 elements ahead = 64 bytes)
                _mm_prefetch(reinterpret_cast<const char*>(neighbors_ptr + current_scan_idx + 16), _MM_HINT_T0);
                
                    _mm_prefetch(reinterpret_cast<const char*>(ll_ptr + current_scan_idx + 16), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(lu_ptr + current_scan_idx + 16), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(rl_ptr + current_scan_idx + 16), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(ru_ptr + current_scan_idx + 16), _MM_HINT_T0);
                

                // Load neighbors
                __m128i v_nbr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(neighbors_ptr + current_scan_idx));

                // Check if any neighbor > right_u (break condition)
                // Unsigned comparison trick: (a ^ sign) > (b ^ sign)
                __m128i v_nbr_adj = _mm_xor_si128(v_nbr, sign_bit);
                __m128i v_break_cmp = _mm_cmpgt_epi32(v_nbr_adj, v_right_u_adj);
                
                // If any bit is set in v_break_cmp, it means at least one neighbor > right_u
                if (_mm_movemask_ps(_mm_castsi128_ps(v_break_cmp)) != 0) {
                    break; // Fallback to scalar to handle the break point accurately
                }

                // Load range attributes
                __m128i v_ll = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ll_ptr + current_scan_idx));
                __m128i v_lu = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lu_ptr + current_scan_idx));
                __m128i v_rl = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rl_ptr + current_scan_idx));
                __m128i v_ru = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ru_ptr + current_scan_idx));

                    // Adjust for unsigned comparison
                    v_ll = _mm_xor_si128(v_ll, sign_bit);
                    v_lu = _mm_xor_si128(v_lu, sign_bit);
                    v_rl = _mm_xor_si128(v_rl, sign_bit);
                    v_ru = _mm_xor_si128(v_ru, sign_bit);

                    // Check conditions:
                    // 1. neighbor >= left_u is guaranteed by lower_bound

                    // 2. left_lower <= left_u  => !(left_lower > left_u) => !(v_ll > v_left_u_adj)
                    __m128i c1_fail = _mm_cmpgt_epi32(v_ll, v_left_u_adj);

                    // 3. left_u <= left_upper  => !(left_u > left_upper) => !(v_left_u_adj > v_lu)
                    __m128i c2_fail = _mm_cmpgt_epi32(v_left_u_adj, v_lu);

                    // 4. right_lower <= right_u => !(right_lower > right_u) => !(v_rl > v_right_u_adj)
                    __m128i c3_fail = _mm_cmpgt_epi32(v_rl, v_right_u_adj);

                    // 5. right_u <= right_upper => !(right_u > right_upper) => !(v_right_u_adj > v_ru)
                    __m128i c4_fail = _mm_cmpgt_epi32(v_right_u_adj, v_ru);

                    // Combine failures
                    __m128i any_fail = _mm_or_si128(c1_fail, c2_fail);
                    any_fail = _mm_or_si128(any_fail, _mm_or_si128(c3_fail, c4_fail));

                // mask = 1 where valid (any_fail is 0)
                int fail_mask = _mm_movemask_ps(_mm_castsi128_ps(any_fail));
                int valid_mask = (~fail_mask) & 0xF;

                while (valid_mask) {
                    int bit = __builtin_ctz(valid_mask);
                    unsigned neighbor = neighbors_ptr[current_scan_idx + bit];
                    
                    // Check visited status
                    if (visited_array[neighbor] != visited_array_tag) {
                        fetched_nns.push_back(neighbor);
                        _mm_prefetch(reinterpret_cast<const char*>(data_wrapper->nodes[neighbor]), _MM_HINT_T0);
                    }
                    
                    valid_mask &= (valid_mask - 1);
                }
            }

        }
        // Scalar Loop for remaining items or after break
        for (; current_scan_idx < end_idx; ++current_scan_idx) {
            const unsigned neighbor = neighbors_ptr[current_scan_idx];

            // neighbor > right_u: break
            if (neighbor > right_u) {
                break;
            }

            if (!skip_range_envelope) {
                if (!((ll_ptr[current_scan_idx] <= left_u && left_u <= lu_ptr[current_scan_idx]) &&
                      (rl_ptr[current_scan_idx] <= right_u && right_u <= ru_ptr[current_scan_idx]))) {
                    continue;
                }
            }
            
            if (visited_array[neighbor] == visited_array_tag) {
                continue;
            }
            fetched_nns.push_back(neighbor);
            _mm_prefetch(reinterpret_cast<const char*>(data_wrapper->nodes[neighbor]), _MM_HINT_T0);
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
    
    // Save flattened SoA arrays
    size_t node_count = row_offset_.size() - 1;
    out.write(reinterpret_cast<const char *>(&node_count), sizeof(node_count));
    
    // Write row_offset
    out.write(reinterpret_cast<const char *>(row_offset_.data()), row_offset_.size() * sizeof(std::size_t));
    
    size_t num_edges = neighbors_.size();
    // Write total edges just in case (though implicit)
    out.write(reinterpret_cast<const char *>(&num_edges), sizeof(num_edges));
    
    // Write arrays
    if (num_edges > 0) {
        out.write(reinterpret_cast<const char *>(neighbors_.data()), num_edges * sizeof(unsigned));
        out.write(reinterpret_cast<const char *>(left_lower_.data()), num_edges * sizeof(unsigned));
        out.write(reinterpret_cast<const char *>(left_upper_.data()), num_edges * sizeof(unsigned));
        out.write(reinterpret_cast<const char *>(right_lower_.data()), num_edges * sizeof(unsigned));
        out.write(reinterpret_cast<const char *>(right_upper_.data()), num_edges * sizeof(unsigned));
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

    row_offset_.resize(node_count + 1);
    in.read(reinterpret_cast<char *>(row_offset_.data()), row_offset_.size() * sizeof(std::size_t));

    size_t num_edges = 0;
    in.read(reinterpret_cast<char *>(&num_edges), sizeof(num_edges));

    neighbors_.resize(num_edges);
    left_lower_.resize(num_edges);
    left_upper_.resize(num_edges);
    right_lower_.resize(num_edges);
    right_upper_.resize(num_edges);

    if (num_edges > 0) {
        in.read(reinterpret_cast<char *>(neighbors_.data()), num_edges * sizeof(unsigned));
        in.read(reinterpret_cast<char *>(left_lower_.data()), num_edges * sizeof(unsigned));
        in.read(reinterpret_cast<char *>(left_upper_.data()), num_edges * sizeof(unsigned));
        in.read(reinterpret_cast<char *>(right_lower_.data()), num_edges * sizeof(unsigned));
        in.read(reinterpret_cast<char *>(right_upper_.data()), num_edges * sizeof(unsigned));
    }

    // Report index-only memory footprint (exclude raw vector storage).
    const std::size_t offset_bytes = row_offset_.capacity() * sizeof(std::size_t);
    const std::size_t neighbor_arrays_bytes =
        neighbors_.capacity() * sizeof(unsigned) +
        left_lower_.capacity() * sizeof(unsigned) +
        left_upper_.capacity() * sizeof(unsigned) +
        right_lower_.capacity() * sizeof(unsigned) +
        right_upper_.capacity() * sizeof(unsigned);
    const std::size_t index_bytes = offset_bytes + neighbor_arrays_bytes;
    const double index_mib = static_cast<double>(index_bytes) / (1024.0 * 1024.0);
    std::cout << "[DSG] Index memory footprint (excluding vectors): "
              << index_mib << " MiB (" << index_bytes << " bytes)" << std::endl;

    edges_amount = num_edges;
    avg_forward_nns = (row_offset_.empty() || node_count == 0)
                          ? 0.0F
                          : static_cast<float>(edges_amount) / static_cast<float>(node_count);
    avg_reverse_nns = 0.0F;
}

void DynamicSegmentGraph::getStats() {
    // calculate the average number of forward neighbors
    edges_amount = neighbors_.size();
    size_t node_count = row_offset_.empty() ? 0 : row_offset_.size() - 1;
    avg_forward_nns = (node_count == 0)
                          ? 0.0F
                          : static_cast<float>(edges_amount) / static_cast<float>(node_count);
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

    // This KNN is for a point that is already inside the temporary HNSW.
    // We can skip the usual top-layer navigation and search only on level-0,
    // using the point's own level-0 neighbors as the initial frontier.
    //
    // Important details for DSG build:
    // - We mark `label` as visited so it never appears in results.
    // - We seed `candidate_set` with `label`'s level-0 neighbors (NOT `label` itself).
    // - We still keep "return all candidates ever in top_candidates" semantics by
    //   storing candidates removed due to ef trimming and re-adding them at the end.

    const hnswlib::tableint ep_id = static_cast<hnswlib::tableint>(label);

    const void *query = static_cast<const void *>(data_wrapper->nodes[label]);

    hnswlib::VisitedList *vl = temp_hnsw_->visited_list_pool_->getFreeVisitedList();
    hnswlib::vl_type *visited_array = vl->mass;
    const hnswlib::vl_type visited_array_tag = vl->curV;

    visited_array[ep_id] = visited_array_tag;  // exclude the query node itself

    using InternalId = hnswlib::tableint;
    using InternalCandidate = std::pair<DistType, InternalId>;
    using CompareByFirst = typename HnswType::CompareByFirst;

    // Reuse scratch buffers to avoid per-label allocations.
    auto &top_candidates = knn_scratch_.top_candidates_heap;
    auto &candidate_set = knn_scratch_.candidate_set_heap;
    auto &removed_candidates = knn_scratch_.removed_candidates;
    top_candidates.clear();
    candidate_set.clear();
    removed_candidates.clear();
    if (top_candidates.capacity() < ef_limit) {
        top_candidates.reserve(ef_limit);
    }
    if (candidate_set.capacity() < ef_limit) {
        candidate_set.reserve(ef_limit);
    }
    if (removed_candidates.capacity() < ef_limit) {
        removed_candidates.reserve(ef_limit);
    }
    const CompareByFirst heap_comp{};

    auto heap_push = [&](std::vector<InternalCandidate> &heap,
                         const InternalCandidate &value) {
        heap.push_back(value);
        std::push_heap(heap.begin(), heap.end(), heap_comp);
    };
    auto heap_pop = [&](std::vector<InternalCandidate> &heap) -> InternalCandidate {
        std::pop_heap(heap.begin(), heap.end(), heap_comp);
        InternalCandidate value = heap.back();
        heap.pop_back();
        return value;
    };

    DistType lower_bound = std::numeric_limits<DistType>::max();

    auto distance_to = [&](InternalId internal_id) -> DistType {
        return temp_hnsw_->fstdistfunc_(query,
                                       temp_hnsw_->getDataByInternalId(internal_id),
                                       temp_hnsw_->dist_func_param_);
    };

    // Seed the frontier with entry-point's level-0 neighbors.
    int *seed_data = reinterpret_cast<int *>(temp_hnsw_->get_linklist0(ep_id));
    const std::size_t seed_size =
        static_cast<std::size_t>(temp_hnsw_->getListCount(reinterpret_cast<hnswlib::linklistsizeint *>(seed_data)));

#ifdef USE_SSE
    _mm_prefetch(reinterpret_cast<const char *>(visited_array + *(seed_data + 1)), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char *>(visited_array + *(seed_data + 1) + 64), _MM_HINT_T0);
    _mm_prefetch(temp_hnsw_->data_level0_memory_ + (*(seed_data + 1)) * temp_hnsw_->size_data_per_element_ +
                     temp_hnsw_->offsetData_,
                 _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char *>(seed_data + 2), _MM_HINT_T0);
#endif

    for (std::size_t j = 1; j <= seed_size; ++j) {
#ifdef USE_SSE
        _mm_prefetch(reinterpret_cast<const char *>(visited_array + *(seed_data + j + 1)), _MM_HINT_T0);
        _mm_prefetch(temp_hnsw_->data_level0_memory_ + (*(seed_data + j + 1)) * temp_hnsw_->size_data_per_element_ +
                         temp_hnsw_->offsetData_,
                     _MM_HINT_T0);
#endif
        const InternalId cand_id = static_cast<InternalId>(*(seed_data + j));
        if (visited_array[cand_id] == visited_array_tag) {
            continue;
        }
        visited_array[cand_id] = visited_array_tag;

        const DistType dist = distance_to(cand_id);
        heap_push(candidate_set, InternalCandidate{-dist, cand_id});
#ifdef USE_SSE
        _mm_prefetch(temp_hnsw_->data_level0_memory_ + candidate_set.front().second * temp_hnsw_->size_data_per_element_ +
                         temp_hnsw_->offsetLevel0_,
                     _MM_HINT_T0);
#endif
        heap_push(top_candidates, InternalCandidate{dist, cand_id});

        if (top_candidates.size() > ef_limit) {
            removed_candidates.emplace_back(heap_pop(top_candidates));
        }
        if (!top_candidates.empty()) {
            lower_bound = top_candidates.front().first;
        }
    }

    // Standard HNSW base-layer best-first traversal (level 0).
    while (!candidate_set.empty()) {
        const auto &current_node_pair = candidate_set.front();
        const DistType candidate_dist = -current_node_pair.first;
        if (candidate_dist > lower_bound) {
            break;
        }
        const auto current_node_pair_value = heap_pop(candidate_set);

        const InternalId current_node_id = current_node_pair_value.second;
        int *data = reinterpret_cast<int *>(temp_hnsw_->get_linklist0(current_node_id));
        const std::size_t size =
            static_cast<std::size_t>(temp_hnsw_->getListCount(reinterpret_cast<hnswlib::linklistsizeint *>(data)));

#ifdef USE_SSE
        _mm_prefetch(reinterpret_cast<const char *>(visited_array + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char *>(visited_array + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(temp_hnsw_->data_level0_memory_ + (*(data + 1)) * temp_hnsw_->size_data_per_element_ +
                         temp_hnsw_->offsetData_,
                     _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char *>(data + 2), _MM_HINT_T0);
#endif

        for (std::size_t j = 1; j <= size; ++j) {
#ifdef USE_SSE
            _mm_prefetch(reinterpret_cast<const char *>(visited_array + *(data + j + 1)), _MM_HINT_T0);
            _mm_prefetch(temp_hnsw_->data_level0_memory_ + (*(data + j + 1)) * temp_hnsw_->size_data_per_element_ +
                             temp_hnsw_->offsetData_,
                         _MM_HINT_T0);
#endif
            const InternalId cand_id = static_cast<InternalId>(*(data + j));
            if (visited_array[cand_id] == visited_array_tag) {
                continue;
            }
            visited_array[cand_id] = visited_array_tag;

            const DistType dist = distance_to(cand_id);
            if (top_candidates.size() < ef_limit || lower_bound > dist) {
                heap_push(candidate_set, InternalCandidate{-dist, cand_id});
#ifdef USE_SSE
                _mm_prefetch(temp_hnsw_->data_level0_memory_ +
                                 candidate_set.front().second * temp_hnsw_->size_data_per_element_ +
                                 temp_hnsw_->offsetLevel0_,
                             _MM_HINT_T0);
#endif
                heap_push(top_candidates, InternalCandidate{dist, cand_id});

                if (top_candidates.size() > ef_limit) {
                    removed_candidates.emplace_back(heap_pop(top_candidates));
                }
                lower_bound = top_candidates.front().first;
            }
        }
    }

    // Re-add any candidates that were trimmed due to ef_limit (ReturnAll semantics).
    for (const auto &removed : removed_candidates) {
        heap_push(top_candidates, removed);
    }

    temp_hnsw_->visited_list_pool_->releaseVisitedList(vl);

    // `top_candidates` is a max-heap by distance (farthest-first pops).
    // Fill output from back to front so `candidates` becomes nearest-first.
    const std::size_t out_size = top_candidates.size();
    candidates.resize(out_size);
    std::size_t out_pos = out_size;
    while (!top_candidates.empty()) {
        const auto [dist, internal_id] = heap_pop(top_candidates);

        const unsigned neighbor_label =
            static_cast<unsigned>(temp_hnsw_->getExternalLabel(internal_id));
        candidates[--out_pos] = {neighbor_label, dist};
    }
}

void DynamicSegmentGraph::applyDfsCompression(
    unsigned center_label,
    std::vector<std::pair<unsigned, DistType>> &candidates) {
    const std::size_t max_neighbors = static_cast<std::size_t>(M);
    const auto &nodes = data_wrapper->nodes;
    const float *const nodes_base = nodes.data();
    const std::size_t nodes_dim = nodes.dim();
    const DistType inv_alpha = static_cast<DistType>(1.0f / alpha);

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
    
    // Resize and clear the domination grid (0 = unknown)
    const size_t grid_size = candidate_count * candidate_count;
    if (dfs_scratch_.domination_grid.size() < grid_size) {
        dfs_scratch_.domination_grid.resize(grid_size);
    }
    // We only need to clear the relevant part, but given the logic,
    // candidates change every time, so we must reset validity.
    // std::fill is fast enough for typical ef sizes (e.g. 100-500 -> 10KB-250KB).
    // Optimization: only clear if we are reusing a large buffer? 
    // For now, simpler to just clear the needed portion.
    std::fill(dfs_scratch_.domination_grid.begin(), dfs_scratch_.domination_grid.begin() + grid_size, 0);

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
            const DistType dom_threshold = candidate_dist * inv_alpha;
            if (candidate_label < L || candidate_label > R) {
                continue;
            }

            const float *const cand_vec =
                nodes_base + static_cast<std::size_t>(candidate_label) * nodes_dim;

            bool dominated = false;
            for (unsigned prev_idx : prefix) {
                // Flat 2D index
                const size_t grid_idx = static_cast<size_t>(prev_idx) * candidate_count + idx;
                const uint8_t status = dfs_scratch_.domination_grid[grid_idx];
                
                if (status != 0) {
                    dominated = (status == 1);
                } else {
                    const unsigned prev_label = ordered[prev_idx].first;
                    const float *const prev_vec =
                        nodes_base + static_cast<std::size_t>(prev_label) * nodes_dim;
                    const DistType pair_dist = dist_func_(
                        cand_vec,
                        prev_vec,
                        dist_func_param_);
                    dominated = pair_dist < dom_threshold;
                    dfs_scratch_.domination_grid[grid_idx] = dominated ? 1 : 2;
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

} // namespace dsg
