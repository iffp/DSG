#include "reader.h"

#include <assert.h>
#include <algorithm>
#include <cstdint>

using std::cout;
using std::endl;
using std::getline;
using std::ifstream;
using std::ios;
using std::string;
using std::vector;

// The .bin format is: [uint32 num_vectors][uint32 dimension][num_vectors * dimension floats]
// All values are little-endian float32, densely packed row-major.
void ReadBinaryVectors(const std::string &file_path,
                       FlatVectors<float> &output,
                       const int64_t max_vectors) {
    std::ifstream ifs(file_path, std::ios::binary);
    assert(ifs.is_open());

    uint32_t num_vectors = 0;
    uint32_t dimension = 0;
    if (!ifs.read(reinterpret_cast<char *>(&num_vectors), sizeof(uint32_t)) ||
        !ifs.read(reinterpret_cast<char *>(&dimension), sizeof(uint32_t))) {
        std::cerr << "Failed to read metadata from " << file_path << std::endl;
        assert(false);
    }
    if (dimension == 0) {
        std::cerr << "Invalid dimension (0) in " << file_path << std::endl;
        assert(false);
    }

    const int64_t num_vectors_ll = static_cast<int64_t>(num_vectors);
    int64_t target = num_vectors_ll;
    if (max_vectors >= 0) {
        target = std::min(max_vectors, num_vectors_ll);
        if (max_vectors > num_vectors_ll) {
            std::cerr << "Requested " << max_vectors << " vectors from " << file_path
                      << " but only " << num_vectors << " available. Loading full file."
                      << std::endl;
        }
    }

    const uint32_t vectors_to_load = static_cast<uint32_t>(target);
    const size_t stride = static_cast<size_t>(dimension) * sizeof(float);

    output.resize(vectors_to_load, dimension);

    const size_t bytes_to_read = static_cast<size_t>(vectors_to_load) * stride;
    char *dest = reinterpret_cast<char *>(output.data());
    if (!ifs.read(dest, bytes_to_read)) {
        std::cerr << "Failed to read any vectors from " << file_path << std::endl;
        output.clear();
        return;
    }
}


// load data and querys
void ReadDataWrapper(std::string &dataset_path,
                     FlatVectors<float> &raw_data,
                     int data_size,
                     std::string &query_path,
                     FlatVectors<float> &querys,
                     int query_size) {
    raw_data.clear();

    // Files must follow the same .bin structure described above.
    ReadBinaryVectors(dataset_path, raw_data, data_size);
    ReadBinaryVectors(query_path, querys, query_size);

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
        vector<string> ret;
        Split(bline, delim, &ret);
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
    }
}
