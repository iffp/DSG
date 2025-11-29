#include "reader.h"

#include <assert.h>

using std::cout;
using std::endl;
using std::getline;
using std::ifstream;
using std::ios;
using std::string;
using std::vector;

std::vector<std::vector<float>> ReadTopN(std::string filename, std::string ext, int top_n) {
    if (ext != "fvecs") {
        std::cerr << "Error: unsupported ext type: " << ext << " in ReadTopN" << std::endl;
        assert(false);
    }

    std::ifstream ifs(filename, std::ios::binary);
    assert(ifs.is_open());

    std::vector<std::vector<float>> vecs;
    if (top_n != -1) {
        vecs.reserve(top_n);
    }

    int D = 0;
    while ((top_n == -1 || static_cast<int>(vecs.size()) < top_n) &&
           ifs.read(reinterpret_cast<char *>(&D), sizeof(int))) {
        std::vector<float> vec(D);
        if (!ifs.read(reinterpret_cast<char *>(vec.data()), sizeof(float) * D)) {
            break;
        }
        vecs.emplace_back(std::move(vec));
    }
    return vecs;
}

/// @brief Reading binary data vectors. Raw data store as a (N x 100)
/// binary file.
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
/// @param N Reading top N vectors
/// @param num_dimensions dimension of dataset
void ReadFvecsTopN(const std::string &file_path,
                   std::vector<std::vector<float>> &data,
                   const uint32_t N,
                   const int num_dimensions) {
    std::cout << "Reading Data: " << file_path << std::endl;
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());

    data.resize(N);
    std::vector<double> buff(num_dimensions);
    int counter = 0;
    while ((counter < N) && (ifs.read((char *)buff.data(), num_dimensions * sizeof(double)))) {
        std::vector<float> row(num_dimensions);
        for (int d = 0; d < num_dimensions; d++) {
            row[d] = static_cast<float>(buff[d]);
        }
        data[counter++] = std::move(row);
    }

    ifs.close();
    std::cout << "Finish Reading Data" << endl;
}


// load data and querys
void ReadDataWrapper(const std::string &dataset, std::string &dataset_path, std::vector<std::vector<float>> &raw_data, const int data_size, std::string &query_path, std::vector<std::vector<float>> &querys, const int query_size, std::vector<int> &search_keys) {
    (void)search_keys;
    raw_data.clear();

    if (dataset == "deep") {
        raw_data = ReadTopN(dataset_path, "fvecs", data_size);
        querys = ReadTopN(query_path, "fvecs", query_size);
    } else if (dataset == "yt8m-video") {
        ReadFvecsTopN(dataset_path, raw_data, data_size, 1024);
        ReadFvecsTopN(query_path, querys, query_size, 1024);
    } else if (dataset == "wiki-image") {
        ReadFvecsTopN(dataset_path, raw_data, data_size, 2048);
        ReadFvecsTopN(query_path, querys, query_size, 2048);
    } else {
        std::cerr << "Unsupported dataset: " << dataset << endl;
        assert(false);
    }

    if (raw_data.size() < data_size) {
        std::cerr << "Dataset Size not reach " << data_size << " the size: " << raw_data.size() << endl;
    }
}

void Split(std::string &s, std::string &delim, std::vector<std::string> *ret) {
    size_t last = 0;
    size_t index = s.find_first_of(delim, last);
    while (index != std::string::npos) {
        ret->push_back(s.substr(last, index - last));
        last = index + 1;
        index = s.find_first_of(delim, last);
    }
    if (index - last > 0) {
        ret->push_back(s.substr(last, index - last));
    }
}

void ReadGroundtruthQuery(vector<vector<int>> &gt,
                          vector<std::pair<int, int>> &query_ranges,
                          vector<int> &query_ids,
                          string gt_path) {
    ifstream infile;
    string bline;
    string delim = ",";
    string space_delim = " ";

    int numCols = 0;
    infile.open(gt_path, ios::in);
    assert(infile.is_open());

    int counter = 0;
    while (getline(infile, bline, '\n')) {
        counter++;
        vector<int> one_gt;
        std::pair<int, int> one_range;
        int one_id;
        vector<string> ret;
        Split(bline, delim, &ret);
        one_id = std::stoi(ret[0]);
        one_range.first = std::stoi(ret[1]);
        one_range.second = std::stoi(ret[2]);
        vector<string> str_gt;
        Split(ret[7], space_delim, &str_gt);
        str_gt.pop_back();
        for (auto ele : str_gt) {
            one_gt.emplace_back(std::stoi(ele));
        }
        gt.emplace_back(one_gt);
        query_ranges.emplace_back(one_range);
        query_ids.emplace_back(one_id);
    }
}
