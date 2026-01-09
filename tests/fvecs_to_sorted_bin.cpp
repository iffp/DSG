/**
 * @file fvecs_to_sorted_bin.cpp
 * @brief Convert .fvecs file to DSG binary format with sorting by attributes.
 * 
 * DSG assumes that vectors are pre-sorted by their attribute values in ascending order.
 * This utility reads vectors from .fvecs format, sorts them by attributes from a CSV file,
 * and writes them in DSG's binary format: [uint32 num][uint32 dim][floats...]
 * 
 * It also creates a mapping file (.mapping) that maps sorted indices to original indices,
 * which is needed to translate query results back to the original ID space for 
 * recall calculation against groundtruth.
 * 
 * Usage: fvecs_to_sorted_bin <input.fvecs> <attributes.csv> <output.bin>
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdint>

/**
 * @brief Read vectors from .fvecs format.
 */
std::vector<std::vector<float>> read_fvecs(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << "\n";
        exit(1);
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
 * @brief Read CSV attributes (one integer per line).
 */
std::vector<int> read_attributes_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << "\n";
        exit(1);
    }
    std::vector<int> attributes;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            attributes.push_back(std::stoi(line));
        }
    }
    file.close();
    return attributes;
}

/**
 * @brief Write binary format: [uint32 num][uint32 dim][floats...]
 * This is the format expected by DSG's DataWrapper.
 */
void write_bin(const std::string& filename, const std::vector<std::vector<float>>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << " for writing\n";
        exit(1);
    }
    
    uint32_t num_points = static_cast<uint32_t>(data.size());
    uint32_t dim = data.empty() ? 0 : static_cast<uint32_t>(data[0].size());
    
    file.write(reinterpret_cast<const char*>(&num_points), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));
    
    for (const auto& vec : data) {
        file.write(reinterpret_cast<const char*>(vec.data()), dim * sizeof(float));
    }
    
    file.close();
}

/**
 * @brief Write mapping file: sorted_index -> original_index
 * Format: [int num_points][size_t mapping[0]]...[size_t mapping[num_points-1]]
 */
void write_mapping(const std::string& filename, const std::vector<size_t>& mapping) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << " for writing\n";
        exit(1);
    }
    
    int num_points = static_cast<int>(mapping.size());
    file.write(reinterpret_cast<const char*>(&num_points), sizeof(int));
    file.write(reinterpret_cast<const char*>(mapping.data()), num_points * sizeof(size_t));
    
    file.close();
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input.fvecs> <attributes.csv> <output.bin>\n";
        std::cerr << "  Converts .fvecs to DSG .bin format and sorts by attributes\n";
        std::cerr << "  Also creates <output.bin>.mapping for ID translation\n";
        return 1;
    }

    std::string input_fvecs = argv[1];
    std::string input_attributes = argv[2];
    std::string output_bin = argv[3];
    std::string output_mapping = output_bin + ".mapping";

    std::cout << "Reading vectors from " << input_fvecs << "...\n";
    std::vector<std::vector<float>> vectors = read_fvecs(input_fvecs);
    
    std::cout << "Reading attributes from " << input_attributes << "...\n";
    std::vector<int> attributes = read_attributes_csv(input_attributes);
    
    if (vectors.size() != attributes.size()) {
        std::cerr << "Error: Number of vectors (" << vectors.size() 
                  << ") does not match number of attributes (" << attributes.size() << ")\n";
        return 1;
    }

    std::cout << "Sorting " << vectors.size() << " vectors by attributes...\n";
    
    // Create index array to track original positions
    std::vector<size_t> indices(vectors.size());
    for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }
    
    // Sort indices based on attributes
    std::sort(indices.begin(), indices.end(), [&attributes](size_t i1, size_t i2) {
        return attributes[i1] < attributes[i2];
    });
    
    // Create sorted vectors and mapping
    std::vector<std::vector<float>> sorted_vectors(vectors.size());
    std::vector<size_t> sorted_to_original(vectors.size());  // mapping[sorted_idx] = original_idx
    
    for (size_t i = 0; i < indices.size(); i++) {
        sorted_vectors[i] = vectors[indices[i]];
        sorted_to_original[i] = indices[i];
    }
    
    std::cout << "Writing sorted vectors to " << output_bin << "...\n";
    write_bin(output_bin, sorted_vectors);
    
    std::cout << "Writing ID mapping to " << output_mapping << "...\n";
    write_mapping(output_mapping, sorted_to_original);
    
    std::cout << "Conversion completed successfully!\n";
    std::cout << "  Vectors: " << vectors.size() << "\n";
    std::cout << "  Dimension: " << (vectors.empty() ? 0 : vectors[0].size()) << "\n";
    
    return 0;
}
