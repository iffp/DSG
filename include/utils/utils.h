/**
 * @file utils.h
 * @brief 提供了一系列实用工具函数。
 */

#pragma once

#include <assert.h>
#include <cstddef>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <sys/time.h>

#ifdef __linux__
#include "sys/sysinfo.h"
#include "sys/types.h"
#elif __APPLE__
#include <mach/mach_host.h>
#include <mach/mach_init.h>
#include <mach/mach_types.h>
#include <mach/vm_statistics.h>
#endif

#include "flat_vectors.h"

// 使用标准库中的命名空间元素
using std::cout;
using std::endl;
using std::getline;
using std::ifstream;
using std::ios;
using std::make_pair;
using std::pair;
using std::string;
using std::vector;

float EuclideanDistance(const float *lhs, const float *rhs, size_t dim);

/**
 * 积累时间差值。
 *
 * @param t2 结束时间。
 * @param t1 开始时间。
 * @param val_time 时间差值。
 */
void AccumulateTime(timeval &t2, timeval &t1, double &val_time);

/**
 * 计算并记录时间差值。
 *
 * @param t1 开始时间。
 * @param t2 结束时间。
 * @param val_time 时间差值。
 */
void CountTime(timeval &t1, timeval &t2, double &val_time);

/**
 * 返回两次时间测量的时间差值。
 *
 * @param t1 开始时间。
 * @param t2 结束时间。
 * @return 时间差值。
 */
double CountTime(timeval &t1, timeval &t2);

template <typename T>
vector<int> sort_indexes(const vector<T> &v) {
    // initialize original index locations
    vector<int> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    stable_sort(idx.begin(), idx.end(),
                [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

    return idx;
}

void WriteVectorToFile(const std::string &file_path, const std::vector<int> &vec);

std::vector<std::vector<unsigned>> ReadAndSplit(const std::string &file_path, int part_num);

template <typename T>
void print_set(const vector<T> &v) {
    if (v.size() == 0) {
        cout << "ERROR: EMPTY VECTOR!" << endl;
        return;
    }
    cout << "vertex in set: {";
    for (size_t i = 0; i < v.size() - 1; i++) {
        cout << v[i] << ", ";
    }
    cout << v.back() << "}" << endl;
}

void logTime(timeval &begin, timeval &end, const string &log);

/**
 * Compute recall between truth and prediction vectors.
 *
 * @param truth Ground-truth ids.
 * @param pred Predicted ids.
 * @return Recall value in [0, 1].
 */
double countRecall(const vector<int> &truth, const vector<int> &pred);
double countRecall(const int *truth, size_t truth_len, const vector<int> &pred);

/**
 * 计算近似比率。
 *
 * @param raw_data 原始数据集。
 * @param truth 真实结果。
 * @param pred 预测结果。
 * @param query 查询点。
 * @return 近似比率。
 */

#define _INT_MAX 2147483640

/**
 * 贪婪算法寻找最近邻。
 *
 * @param dpts 数据点集。
 * @param query 查询点。
 * @param k_smallest 寻找的最小数量。
 * @return 最近邻点的索引列表。
 */
size_t greedyNearestInto(const FlatVectors<float> &dpts,
                         const float *query,
                         const int l_bound,
                         const int r_bound,
                         const int k_smallest,
                         int *output);
vector<int> scanNearest(const FlatVectors<float> &dpts,
                        const vector<int> &keys,
                        const float *query,
                        const int l_bound,
                        const int r_bound,
                        const int k_smallest);