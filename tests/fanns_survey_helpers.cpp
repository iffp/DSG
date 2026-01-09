/**
 * @file fanns_survey_helpers.cpp
 * @brief Helper functions for the FANNS benchmarking suite integration.
 * 
 * These utilities read standard file formats (.fvecs, .ivecs, .csv) and 
 * provide memory/thread monitoring for benchmarking purposes.
 */

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <thread>
#include <atomic>
#include <chrono>
#include <unistd.h>
#include <utility>
#include <set>

#include "global_thread_counter.h"

/**
 * @brief Read vectors from .fvecs format file.
 * @param filename Path to the .fvecs file
 * @return Vector of vectors containing the data
 */
std::vector<std::vector<float>> read_fvecs(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file for reading: " << filename << "\n";
        return {};
    }
    std::vector<std::vector<float>> dataset;
    while (file) {
        int d;
        if (!file.read(reinterpret_cast<char*>(&d), sizeof(int))) break;
        std::vector<float> vec(d);
        if (!file.read(reinterpret_cast<char*>(vec.data()), d * sizeof(float))) break;
        dataset.push_back(std::move(vec));
    }
    file.close();
    return dataset;
}

/**
 * @brief Read vectors from .ivecs format file.
 * @param filename Path to the .ivecs file
 * @return Vector of vectors containing integer data (e.g., groundtruth IDs)
 */
std::vector<std::vector<int>> read_ivecs(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file for reading: " << filename << "\n";
        return {};
    }
    std::vector<std::vector<int>> dataset;
    while (file) {
        int d;
        if (!file.read(reinterpret_cast<char*>(&d), sizeof(int))) break;
        std::vector<int> vec(d);
        if (!file.read(reinterpret_cast<char*>(vec.data()), d * sizeof(int))) break;
        dataset.push_back(std::move(vec));
    }
    file.close();
    return dataset;
}

/**
 * @brief Read one integer per line from a text file.
 * @param filename Path to the text file
 * @return Vector of integers
 */
std::vector<int> read_one_int_per_line(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }
    std::vector<int> result;
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        std::stringstream ss(line);
        int value;
        if (!(ss >> value)) {
            throw std::runtime_error("Non-integer or empty line at line " + std::to_string(line_number));
        }
        std::string extra;
        if (ss >> extra) {
            throw std::runtime_error("More than one value on line " + std::to_string(line_number));
        }
        result.push_back(value);
    }
    return result;
}

/**
 * @brief Read multiple comma-separated integers per line.
 * @param filename Path to the text file
 * @return Vector of vectors of integers
 */
std::vector<std::vector<int>> read_multiple_ints_per_line(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }
    std::vector<std::vector<int>> data;
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        std::vector<int> row;
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try {
                if (!token.empty()) {
                    row.push_back(std::stoi(token));
                }
            } catch (...) {
                throw std::runtime_error("Invalid integer on line " + std::to_string(line_number));
            }
        }
        data.push_back(std::move(row));
    }
    return data;
}

/**
 * @brief Read two integers per line in "low-high" format (range queries).
 * @param filename Path to the text file
 * @return Vector of pairs (low, high)
 */
std::vector<std::pair<int, int>> read_two_ints_per_line(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }
    std::vector<std::pair<int, int>> result;
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        std::stringstream ss(line);
        std::string first, second;
        if (!std::getline(ss, first, '-') || !std::getline(ss, second) || !ss.eof()) {
            throw std::runtime_error("Invalid format at line " + std::to_string(line_number));
        }
        try {
            int a = std::stoi(first);
            int b = std::stoi(second);
            result.emplace_back(a, b);
        } catch (...) {
            throw std::runtime_error("Invalid integer value at line " + std::to_string(line_number));
        }
    }
    return result;
}

/**
 * @brief Print peak memory footprint from /proc/self/status.
 */
void peak_memory_footprint() {
    unsigned iPid = (unsigned)getpid();

    std::cout << "PID: " << iPid << std::endl;

    std::string status_file = "/proc/" + std::to_string(iPid) + "/status";
    std::ifstream info(status_file);
    if (!info.is_open()) {
        std::cout << "memory information open error!" << std::endl;
        return;
    }
    std::string tmp;
    while (getline(info, tmp)) {
        if (tmp.find("Name:") != std::string::npos || 
            tmp.find("VmPeak:") != std::string::npos || 
            tmp.find("VmHWM:") != std::string::npos) {
            std::cout << tmp << std::endl;
        }
    }
    info.close();
}

/**
 * @brief Read current thread count from /proc/self/status.
 * @return Current number of threads or -1 on error
 */
int get_thread_count() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("Threads:", 0) == 0) {
            return std::stoi(line.substr(8));
        }
    }
    return -1;
}

/**
 * @brief Background monitor that updates peak thread count.
 * @param done_flag Atomic flag to signal when to stop monitoring
 */
void monitor_thread_count(std::atomic<bool>& done_flag) {
    while (!done_flag) {
        int current = get_thread_count();
        if (current > peak_threads) {
            peak_threads = current;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

/**
 * @brief Calculate recall given groundtruth and results.
 * @param groundtruth Vector of groundtruth IDs
 * @param results Vector of result IDs  
 * @param k Number of top results to consider
 * @return Recall value between 0 and 1
 */
double calculate_recall(const std::vector<int>& groundtruth, 
                        const std::vector<int>& results, 
                        int k) {
    std::set<int> gt_set(groundtruth.begin(), groundtruth.begin() + std::min((int)groundtruth.size(), k));
    int hits = 0;
    for (int i = 0; i < std::min((int)results.size(), k); ++i) {
        if (gt_set.count(results[i])) {
            ++hits;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(k);
}
