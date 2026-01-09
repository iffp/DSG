/**
 * @file fvecs_to_bin.cpp
 * @brief Convert .fvecs file to DSG binary format without sorting.
 * 
 * This utility is used for converting query vectors which don't need to be sorted.
 * It reads vectors from .fvecs format and writes them in DSG's binary format:
 * [uint32 num][uint32 dim][floats...]
 * 
 * Usage: fvecs_to_bin <input.fvecs> <output.bin>
 */

#include <iostream>
#include <fstream>
#include <vector>
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

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.fvecs> <output.bin>\n";
        std::cerr << "  Converts .fvecs to DSG .bin format (no sorting)\n";
        return 1;
    }

    std::string input_fvecs = argv[1];
    std::string output_bin = argv[2];

    std::cout << "Reading vectors from " << input_fvecs << "...\n";
    std::vector<std::vector<float>> vectors = read_fvecs(input_fvecs);
    
    std::cout << "Writing vectors to " << output_bin << "...\n";
    write_bin(output_bin, vectors);
    
    std::cout << "Conversion completed successfully!\n";
    std::cout << "  Vectors: " << vectors.size() << "\n";
    std::cout << "  Dimension: " << (vectors.empty() ? 0 : vectors[0].size()) << "\n";
    
    return 0;
}
