/**
 * @file buildindex_wrapper.cpp
 * @brief Index construction wrapper for DSG integration with FANNS benchmarking suite.
 * 
 * This wrapper builds a DSG index from pre-sorted binary vectors and outputs
 * standardized statistics for the benchmarking suite to parse.
 * 
 * Command line arguments:
 *   --data_path <path>       Path to sorted database vectors (.bin format)
 *   --index_path <path>      Path where index will be saved
 *   --M <int>                Max out-degree parameter (default 16)
 *   --ef_construction <int>  HNSW ef construction parameter (default 100)
 *   --ef_max <int>           Maximum candidates for DFS compression (default 400)
 *   --alpha <float>          Vamana-style pruning parameter (default 1.0)
 * 
 * Output format (parsed by dsg.py):
 *   Build time (s): <time>
 *   Peak thread count: <count>
 *   VmPeak: <value> kB
 *   VmHWM: <value> kB
 */

#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <omp.h>

#include "fanns_survey_helpers.cpp"
#include "global_thread_counter.h"

#include "base_hnsw/hnswlib.h"
#include "data_wrapper.h"
#include "reader.h"
#include "dsg.h"

// Global atomic to store peak thread count
std::atomic<int> peak_threads(1);

int main(int argc, char **argv) {
    // Default parameters matching DSG paper defaults
    std::string data_path;
    std::string index_path;
    unsigned M = 16;
    unsigned ef_construction = 100;
    unsigned ef_max = 400;
    float alpha = 1.0f;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data_path" && i + 1 < argc) {
            data_path = argv[++i];
        } else if (arg == "--index_path" && i + 1 < argc) {
            index_path = argv[++i];
        } else if (arg == "--M" && i + 1 < argc) {
            M = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if (arg == "--ef_construction" && i + 1 < argc) {
            ef_construction = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if (arg == "--ef_max" && i + 1 < argc) {
            ef_max = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if (arg == "--alpha" && i + 1 < argc) {
            alpha = std::stof(argv[++i]);
        }
    }
    
    // Validate required arguments
    if (data_path.empty()) {
        std::cerr << "Error: --data_path is required\n";
        return 1;
    }
    if (index_path.empty()) {
        std::cerr << "Error: --index_path is required\n";
        return 1;
    }

    try {
        // Get number of hardware threads for index construction
        unsigned int nthreads = std::thread::hardware_concurrency();
        if (nthreads == 0) nthreads = 1;
        omp_set_num_threads(nthreads);
        
        std::cout << "[DSG] Building index with parameters:" << std::endl;
        std::cout << "  data_path: " << data_path << std::endl;
        std::cout << "  index_path: " << index_path << std::endl;
        std::cout << "  M: " << M << std::endl;
        std::cout << "  ef_construction: " << ef_construction << std::endl;
        std::cout << "  ef_max: " << ef_max << std::endl;
        std::cout << "  alpha: " << alpha << std::endl;
        std::cout << "  threads: " << nthreads << std::endl;

        // First, read the header of the binary file to get num_vectors
        int data_size = 0;
        {
            std::ifstream ifs(data_path, std::ios::binary);
            if (!ifs) {
                std::cerr << "Error: Cannot open data file: " << data_path << std::endl;
                return 1;
            }
            uint32_t num_vectors = 0;
            ifs.read(reinterpret_cast<char*>(&num_vectors), sizeof(uint32_t));
            data_size = static_cast<int>(num_vectors);
        }
        std::cout << "[DSG] Data file contains " << data_size << " vectors" << std::endl;

        // Create DataWrapper with correct size
        // NOTE: We don't use readData() because it requires BOTH database and query files.
        // Instead, we load data directly using ReadBinaryVectors.
        DataWrapper data_wrapper(0, 10, "custom", data_size);
        
        // Load database vectors directly (bypassing readData which requires query file)
        ReadBinaryVectors(data_path, data_wrapper.nodes, data_size);
        data_wrapper.data_dim = data_wrapper.nodes.dim();
        
        std::cout << "[DSG] Loaded " << data_wrapper.nodes.size() << " vectors of dimension " 
                  << data_wrapper.data_dim << std::endl;

        // Create L2 space and DSG index
        hnswlib::L2Space space(data_wrapper.data_dim);
        dsg::DynamicSegmentGraph index(&space, &data_wrapper);
        
        // Set index parameters
        index.M = M;
        index.ef_construction = ef_construction;
        index.ef_max = ef_max;
        index.alpha = alpha;

        // Start thread monitoring
        std::atomic<bool> done(false);
        std::thread monitor(monitor_thread_count, std::ref(done));

        // Start timing - measure only index construction
        auto start_time = std::chrono::high_resolution_clock::now();

        // Build the index
        index.build();

        // Stop timing
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        // Stop monitoring
        done = true;
        monitor.join();

        // Save the index
        index.save(index_path);
        std::cout << "[DSG] Index saved to " << index_path << std::endl;

        // Print statistics in the expected format for the benchmarking suite
        std::cout << "Index construction completed." << std::endl;
        std::cout << "Build time (s): " << elapsed.count() << std::endl;
        std::cout << "Peak thread count: " << peak_threads.load() << std::endl;
        
        // Print memory footprint
        peak_memory_footprint();

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
