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

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "flat_vectors.h"

// Read vectors stored in the unified binary format (metadata header followed by
// float32 payloads). If max_vectors < 0 the entire file is loaded.
void ReadBinaryVectors(const std::string &file_path,
                       FlatVectors<float> &output,
                       int64_t max_vectors = -1);

void ReadDataWrapper(std::string &dataset_path,
                     FlatVectors<float> &raw_data,
                     int data_size,
                     std::string &query_path,
                     FlatVectors<float> &querys,
                     int query_size);

void ReadGroundtruthQuery(std::vector<std::vector<int>> &gt,
                          std::vector<std::pair<int, int>> &query_ranges,
                          std::string gt_path);