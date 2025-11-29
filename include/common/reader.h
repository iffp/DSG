/**
 * @file reader.h
 * @author Zhencan Peng (zhencan.peng@rutgers.edu)
 * @brief Read vector data
 * @date 2025-11-21
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// Wrapper. Read top-N vectors (currently supports "fvecs" inputs only).
// If top_n = -1, then read all vectors from the file.
std::vector<std::vector<float>> ReadTopN(std::string filename, std::string ext,
                                         int top_n = -1);

void ReadDataWrapper(const std::string &dataset, std::string &dataset_path,
                     std::vector<std::vector<float>> &raw_data,
                     const int data_size, std::string &query_path,
                     std::vector<std::vector<float>> &querys,
                     const int query_size, std::vector<int> &search_keys);

void ReadGroundtruthQuery(std::vector<std::vector<int>> &gt,
                          std::vector<std::pair<int, int>> &query_ranges,
                          std::vector<int> &query_ids, std::string gt_path);