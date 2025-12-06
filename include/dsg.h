// Author: Zhencan Peng, 2025/11/30
/**
 * @file dsg.h
 * @brief Dynamic Segment Graph (DSG) header. Rebuilt from compact_graph.h
 * @details This header declares the Dynamic Segment Graph, an implementation that first builds a temporary HNSW,
 *          runs ef_max-sized neighbor searches for every node, applies a DFS-based compression over the neighbors,
 *          and stores only forward segment edges (reverse edges and insertions will be implemented later).
 *          The class inherits `BaseIndex` so it can plug into the existing indexing/search pipeline while exposing
 *          range-filtering queries over compressed segment neighbors. The file introduces the basic data structures
 *          (segment edges, per-node containers, DFS scratch buffers) together with the public API needed for building,
 *          querying, and persisting the DSG index. Insert/update workflows are intentionally omitted in this version
 *          and will be added after the core rebuild is complete.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <queue>
#include <utility>
#include <vector>
#include <unordered_map>

#include "base_hnsw/hnswalg.h"
#include "data_wrapper.h"
#include "base_index.h"

namespace dsg {

using hnswlib::tableint;

/**
 * @brief A compressed forward edge that captures the valid segment ranges.
 */
template <typename dist_t>
struct SegmentEdge {
    /// Neighbor's external label.
    unsigned external_id = 0;
    /// Inclusive smallest valid left boundary.
    unsigned left_lower = 0;
    /// Inclusive largest valid left boundary.
    unsigned left_upper = 0;
    /// Inclusive smallest valid right boundary.
    unsigned right_lower = 0;
    /// Inclusive largest valid right boundary.
    unsigned right_upper = 0;

    SegmentEdge() = default;
    SegmentEdge(unsigned id,
                unsigned ll,
                unsigned lr,
                unsigned rl,
                unsigned rr) :
        external_id(id),
        left_lower(ll),
        left_upper(lr),
        right_lower(rl),
        right_upper(rr) {
    }

    bool coversRange(unsigned query_left, unsigned query_right) const noexcept {
        return (left_lower <= query_left && query_left <= left_upper) &&
               (right_lower <= query_right && query_right <= right_upper);
    }
};

/**
 * @brief Scratchpad buffers reused during DFS.
 */
struct DfsScratch {
    /// Sorted (id, distance) pairs produced by the temporary HNSW search.
    std::vector<std::pair<unsigned, float>> ordered_candidates;
    /// Current DFS prefix indexes into ordered_candidates.
    std::vector<unsigned> prefix;
    /// Marks whether a candidate has already been accepted as a neighbor.
    std::vector<bool> is_neighbor;
    /// Per-candidate left-lower boundary.
    std::vector<unsigned> left_lower;
    /// Per-candidate left-upper boundary.
    std::vector<unsigned> left_upper;
    /// Per-candidate right-lower boundary.
    std::vector<unsigned> right_lower;
    /// Per-candidate right-upper boundary.
    std::vector<unsigned> right_upper;
    /// Cache of domination checks between candidate pairs.
    std::unordered_map<std::uint64_t, bool> domination_cache;
};

/**
 * @brief Dynamic Segment Graph that relies on a temporary HNSW during build.
 */
class DynamicSegmentGraph : public BaseIndex {
public:
    using DistType = float;
    using HnswType = hnswlib::HierarchicalNSW<DistType>;

    /**
     * @brief Construct an empty DSG with the given data wrapper and L2 space.
     */
    DynamicSegmentGraph(hnswlib::SpaceInterface<DistType> *space,
                        const DataWrapper *data);
    /**
     * @brief Release the temporary HNSW and any scratch buffers.
     */
    ~DynamicSegmentGraph() override;

    /**
     * @brief Build the DSG by first constructing a temporary HNSW then compressing neighbors.
     */
    void build() override;
    /**
     * @brief Execute a range query inside [query_bound.first, query_bound.second].
     *        Results are written into `returned_nns`.
     */
    void rangeSearch(const float *query,
                     const std::pair<int, int> query_bound) override;

    /**
     * @brief Persist the compressed forward edges to disk.
     */
    void save(const std::string &file_path) override;
    /**
     * @brief Load a previously saved DSG from disk.
     */
    void load(const std::string &file_path) override;

    /**
     * @brief Report index statistics to stdout.
     */
    void getStats();
    /// Last query hop count.
    std::size_t last_hop_count() const noexcept { return last_hop_count_; }
    /// Last query distance evaluation count.
    std::size_t last_distance_eval_count() const noexcept { return last_distance_eval_count_; }

private:
    /// Allocate and build the temporary HNSW used for candidate generation.
    void initializeTemporaryHnsw(size_t ef_limit);
    /// Run an ef_max-sized search in the temporary HNSW for the target label.
    void runKnnForLabel(unsigned label,
                        size_t ef_limit,
                        std::vector<std::pair<unsigned, DistType>> &candidates);
    /// Apply DFS-based dominance pruning and produce segment ranges.
    void applyDfsCompression(unsigned center_label,
                             std::vector<std::pair<unsigned, DistType>> &candidates);
    /// Move the compressed neighbors from scratch buffers into forward_edges_.
    void storeForwardEdges(unsigned center_label);

private:
    /// Distance space used to construct the temporary HNSW.
    hnswlib::SpaceInterface<DistType> *space_ = nullptr;
    /// Temporary HNSW instance used only during build.
    std::unique_ptr<HnswType> temp_hnsw_;
    /// Distance function supplied by the space.
    hnswlib::DISTFUNC<DistType> dist_func_ = nullptr;
    /// Parameter blob forwarded to the distance function.
    void *dist_func_param_ = nullptr;
    /// Compressed forward edges for every label.
    std::vector<std::vector<SegmentEdge<DistType>>> forward_edges_;
    /// Reusable buffers for DFS compression.
    DfsScratch dfs_scratch_;
    /// Pool for visited lists
    hnswlib::VisitedListPool *visited_list_pool_ = nullptr;
    /// Last query hop count.
    std::size_t last_hop_count_ = 0;
    /// Last query distance evaluation count.
    std::size_t last_distance_eval_count_ = 0;
};

} // namespace dsg

