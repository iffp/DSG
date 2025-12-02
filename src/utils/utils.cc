#include "utils.h"

float EuclideanDistance(const float *lhs, const float *rhs, size_t dim) {
    float ans = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = lhs[i] - rhs[i];
        ans += diff * diff;
    }
    return ans;
}

// t1:begin, t2:end
void AccumulateTime(timeval &t1, timeval &t2, double &val_time) {
    val_time += (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) * 1.0 / CLOCKS_PER_SEC);
}

void CountTime(timeval &t1, timeval &t2, double &val_time) {
    val_time = 0;
    val_time += (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) * 1.0 / CLOCKS_PER_SEC);
}

double CountTime(timeval &t1, timeval &t2) {
    double val_time = 0.0;
    val_time += (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) * 1.0 / CLOCKS_PER_SEC);
    return val_time;
}

void logTime(timeval &begin, timeval &end, const string &log) {
    gettimeofday(&end, NULL);
    fprintf(stdout, ("# " + log + ": %.7fs\n").c_str(),
            end.tv_sec - begin.tv_sec + (end.tv_usec - begin.tv_usec) * 1.0 / CLOCKS_PER_SEC);
};

double countPrecision(const vector<int> &truth, const vector<int> &pred) {
    double num_right = 0;
    for (auto one : truth) {
        if (std::find(pred.begin(), pred.end(), one) != pred.end()) {
            num_right += 1;
        }
    }
    return num_right / truth.size();
}

double countPrecision(const int *truth,
                      size_t truth_len,
                      const vector<int> &pred) {
    if (truth_len == 0) {
        return 0.0;
    }
    double num_right = 0;
    for (size_t i = 0; i < truth_len; ++i) {
        const int val = truth[i];
        if (std::find(pred.begin(), pred.end(), val) != pred.end()) {
            num_right += 1;
        }
    }
    return num_right / static_cast<double>(truth_len);
}



size_t greedyNearestInto(const FlatVectors<float> &dpts,
                         const float *query,
                         const int l_bound,
                         const int r_bound,
                         const int k_smallest,
                         int *output) {
    std::priority_queue<std::pair<float, int>> top_candidates;
    float lower_bound = _INT_MAX;
    const size_t dim = dpts.dim();
    for (size_t i = l_bound; i <= r_bound; i++) {
        float dist = EuclideanDistance(query, dpts[i], dim);
        if (top_candidates.size() < k_smallest || dist < lower_bound) {
            top_candidates.push(std::make_pair(dist, i));
            if (top_candidates.size() > k_smallest) {
                top_candidates.pop();
            }

            lower_bound = top_candidates.top().first;
        }
    }
    size_t count = 0;
    while (!top_candidates.empty()) {
        output[count++] = top_candidates.top().second;
        top_candidates.pop();
    }
    return count;
}

vector<int> scanNearest(const FlatVectors<float> &dpts,
                        const vector<int> &keys,
                        const float *query,
                        const int l_bound,
                        const int r_bound,
                        const int k_smallest) {
    std::priority_queue<std::pair<float, int>> top_candidates;
    float lower_bound = _INT_MAX;
    const size_t dim = dpts.dim();
    for (size_t i = 0; i < dpts.size(); i++) {
        if (keys[i] >= l_bound && keys[i] <= r_bound) {
            float dist = EuclideanDistance(query, dpts[i], dim);
            if (top_candidates.size() < k_smallest || dist < lower_bound) {
                top_candidates.push(std::make_pair(dist, keys[i]));
                if (top_candidates.size() > k_smallest) {
                    top_candidates.pop();
                }

                lower_bound = top_candidates.top().first;
            }
        }
    }
    vector<int> res;
    while (!top_candidates.empty()) {
        res.emplace_back(top_candidates.top().second);
        top_candidates.pop();
    }
    return res;
}

void WriteVectorToFile(const std::string &file_path, const std::vector<int> &vec) {
    std::ofstream out(file_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + file_path);
    }

    size_t size = vec.size();
    out.write(reinterpret_cast<const char *>(&size), sizeof(size));            // Write vector size
    out.write(reinterpret_cast<const char *>(vec.data()), size * sizeof(int)); // Write vector data
    out.close();
}

std::vector<std::vector<unsigned>> ReadAndSplit(const std::string &file_path, int part_num) {
    if (part_num <= 0) {
        throw std::invalid_argument("part_num must be greater than zero.");
    }

    // Read the entire vector from file
    std::ifstream in(file_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + file_path);
    }

    size_t size;
    in.read(reinterpret_cast<char *>(&size), sizeof(size)); // Read vector size

    std::vector<unsigned> vec(size);
    in.read(reinterpret_cast<char *>(vec.data()), size * sizeof(unsigned)); // Read vector data
    in.close();

    // Split the vector into parts
    std::vector<std::vector<unsigned>> result;
    int part_size = std::ceil(static_cast<double>(size) / part_num);

    for (int i = 0; i < part_num; ++i) {
        auto start = vec.begin() + i * part_size;
        auto end = (i == part_num - 1) ? vec.end() : start + part_size;
        result.emplace_back(start, end);
    }

    return result;
}
