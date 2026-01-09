/**
 * @file search_wrapper.cpp
 * @brief Query execution wrapper for DSG integration with FANNS benchmarking suite.
 * 
 * This wrapper loads a DSG index and executes range-filtered queries, then
 * computes recall against groundtruth and outputs standardized statistics.
 * 
 * Command line arguments:
 *   --data_path <path>          Path to sorted database vectors (.bin format)
 *   --query_path <path>         Path to query vectors (.bin format)
 *   --query_ranges_file <path>  Path to query ranges file (CSV: "low-high" per line)
 *   --groundtruth_file <path>   Path to groundtruth file (.ivecs format)
 *   --index_path <path>         Path to the saved DSG index
 *   --search_ef <int>           Search ef parameter
 *   --k <int>                   Number of neighbors to return (default 10)
 * 
 * Output format (parsed by dsg.py):
 *   Query time (s): <time>
 *   Peak thread count: <count>
 *   QPS: <value>
 *   Recall: <value>
 *   VmPeak: <value> kB
 *   VmHWM: <value> kB
 */

#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <omp.h>
#include <limits>

#include "fanns_survey_helpers.cpp"
#include "global_thread_counter.h"

#include "base_hnsw/hnswlib.h"
#include "data_wrapper.h"
#include "dsg.h"

// Global atomic to store peak thread count
std::atomic<int> peak_threads(1);

int main(int argc, char **argv) {
    // Default parameters
    std::string data_path;
    std::string query_path;
    std::string query_ranges_file;
    std::string groundtruth_file;
    std::string index_path;
    unsigned search_ef = 100;
    int k = 10;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data_path" && i + 1 < argc) {
            data_path = argv[++i];
        } else if (arg == "--query_path" && i + 1 < argc) {
            query_path = argv[++i];
        } else if (arg == "--query_ranges_file" && i + 1 < argc) {
            query_ranges_file = argv[++i];
        } else if (arg == "--groundtruth_file" && i + 1 < argc) {
            groundtruth_file = argv[++i];
        } else if (arg == "--index_path" && i + 1 < argc) {
            index_path = argv[++i];
        } else if (arg == "--search_ef" && i + 1 < argc) {
            search_ef = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if (arg == "--k" && i + 1 < argc) {
            k = std::stoi(argv[++i]);
        }
    }
    
    // Validate required arguments
    if (data_path.empty()) {
        std::cerr << "Error: --data_path is required\n";
        return 1;
    }
    if (query_path.empty()) {
        std::cerr << "Error: --query_path is required\n";
        return 1;
    }
    if (query_ranges_file.empty()) {
        std::cerr << "Error: --query_ranges_file is required\n";
        return 1;
    }
    if (groundtruth_file.empty()) {
        std::cerr << "Error: --groundtruth_file is required\n";
        return 1;
    }
    if (index_path.empty()) {
        std::cerr << "Error: --index_path is required\n";
        return 1;
    }

    try {
        // IMPORTANT: Restrict number of threads to 1 for query execution
        // This is required by the benchmarking protocol
        omp_set_num_threads(1);
        
        std::cout << "[DSG] Query execution with parameters:" << std::endl;
        std::cout << "  data_path: " << data_path << std::endl;
        std::cout << "  query_path: " << query_path << std::endl;
        std::cout << "  query_ranges_file: " << query_ranges_file << std::endl;
        std::cout << "  groundtruth_file: " << groundtruth_file << std::endl;
        std::cout << "  index_path: " << index_path << std::endl;
        std::cout << "  search_ef: " << search_ef << std::endl;
        std::cout << "  k: " << k << std::endl;

        // Read query ranges from CSV file (format: "low-high" per line)
        std::vector<std::pair<int, int>> query_ranges = read_two_ints_per_line(query_ranges_file);
        std::cout << "[DSG] Loaded " << query_ranges.size() << " query ranges" << std::endl;
        
        // Read groundtruth from ivecs file (contains IDs in ORIGINAL unsorted order)
        std::vector<std::vector<int>> groundtruth = read_ivecs(groundtruth_file);
        std::cout << "[DSG] Loaded " << groundtruth.size() << " groundtruth entries" << std::endl;
        
        if (query_ranges.size() != groundtruth.size()) {
            std::cerr << "Error: Number of query ranges (" << query_ranges.size() 
                      << ") does not match number of groundtruth entries (" << groundtruth.size() << ")\n";
            return 1;
        }

        // Load the ID mapping: sorted_index -> original_index
        // This translates from sorted database IDs (used by DSG) to original IDs (used in groundtruth)
        std::string mapping_file = data_path + ".mapping";
        std::ifstream mapping_in(mapping_file, std::ios::binary);
        if (!mapping_in) {
            std::cerr << "Error: Unable to open mapping file: " << mapping_file << std::endl;
            return 1;
        }
        int num_data_points;
        mapping_in.read(reinterpret_cast<char*>(&num_data_points), sizeof(int));
        std::vector<size_t> sorted_to_original(num_data_points);
        mapping_in.read(reinterpret_cast<char*>(sorted_to_original.data()), num_data_points * sizeof(size_t));
        mapping_in.close();
        std::cout << "[DSG] Loaded ID mapping (" << num_data_points << " points)" << std::endl;

        // Read query file header to get query count
        int query_file_count = 0;
        {
            std::ifstream qfs(query_path, std::ios::binary);
            if (!qfs) {
                std::cerr << "Error: Cannot open query file: " << query_path << std::endl;
                return 1;
            }
            uint32_t num_queries = 0;
            qfs.read(reinterpret_cast<char*>(&num_queries), sizeof(uint32_t));
            query_file_count = static_cast<int>(num_queries);
        }
        
        // The actual query count is the minimum of queries in file and query_ranges
        int query_num = std::min(query_file_count, static_cast<int>(query_ranges.size()));
        std::cout << "[DSG] Will process " << query_num << " queries" << std::endl;

        // Load data and queries using DSG's DataWrapper
        DataWrapper data_wrapper(query_num, k, "custom", num_data_points);
        data_wrapper.readData(data_path, query_path);
        
        std::cout << "[DSG] Loaded " << data_wrapper.nodes.size() << " database vectors" << std::endl;
        std::cout << "[DSG] Loaded " << data_wrapper.querys.size() << " query vectors" << std::endl;

        // Create L2 space and DSG index
        hnswlib::L2Space space(data_wrapper.data_dim);
        dsg::DynamicSegmentGraph index(&space, &data_wrapper);
        
        // Set query parameters
        index.setQueryTopK(static_cast<unsigned>(k));
        index.setSearchEf(search_ef);
        
        // Load the index
        index.load(index_path);
        std::cout << "[DSG] Index loaded from " << index_path << std::endl;

        // Store query results for recall calculation
        std::vector<std::vector<int>> query_results(query_num);

        // Start thread monitoring
        std::atomic<bool> done(false);
        std::thread monitor(monitor_thread_count, std::ref(done));

        // Start timing - measure only query execution
        auto start_time = std::chrono::high_resolution_clock::now();

        // Execute queries
        for (int i = 0; i < query_num; i++) {
            const float* query_vec = data_wrapper.querys.at(i);
            auto range_pair = query_ranges[i];
            
            // Execute range search
            index.rangeSearch(query_vec, range_pair);
            
            // Store results - translate from sorted to original ID space
            query_results[i].reserve(k);
            for (size_t j = 0; j < index.returned_nns.size() && j < static_cast<size_t>(k); j++) {
                unsigned sorted_id = index.returned_nns[j];
                if (sorted_id != std::numeric_limits<unsigned>::max() && 
                    sorted_id < sorted_to_original.size()) {
                    int original_id = static_cast<int>(sorted_to_original[sorted_id]);
                    query_results[i].push_back(original_id);
                }
            }
        }

        // Stop timing
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        // Stop monitoring
        done = true;
        monitor.join();

        // Calculate QPS (queries per second)
        double qps = query_num / elapsed.count();

        // Calculate recall AFTER timing stops (exclude from performance measurement)
        double total_recall = 0.0;
        for (int i = 0; i < query_num; i++) {
            total_recall += calculate_recall(groundtruth[i], query_results[i], k);
        }
        double avg_recall = total_recall / query_num;

        // Print statistics in the expected format for the benchmarking suite
        std::cout << "Query execution completed." << std::endl;
        std::cout << "Query time (s): " << elapsed.count() << std::endl;
        std::cout << "Peak thread count: " << peak_threads.load() << std::endl;
        std::cout << "QPS: " << qps << std::endl;
        std::cout << "Recall: " << avg_recall << std::endl;
        
        // Print memory footprint
        peak_memory_footprint();

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
