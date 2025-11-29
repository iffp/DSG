/**
 * @file reader.h
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Read Vector data
 * @date 2023-04-21
 *
 * @copyright Copyright (c) 2023
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