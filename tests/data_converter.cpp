/**
 * @file data_converter.cpp
 * @brief Utility to normalize legacy dataset files into the unified binary
 *        layout expected by DynamicSegmentGraph. The target format stores a
 *        uint32_t vector count followed by a uint32_t dimension, then the raw
 *        float values laid out row-major.
 *
 *        Supported inputs:
 *          1. "fvecs": standard Faiss-style files where every vector is stored
 *             as [int32 dimension][dimension float32 values].
 *          2. "double": raw double-precision arrays with a fixed dimension that
 *             must be provided on the command line.
 *
 *        Examples:
 *          ./data_converter input.fvecs output.bin fvecs
 *          ./data_converter wiki_image_embedding.fvecs wiki_image_embedding.bin double 2048
 *
 *        This tool performs streaming conversion, so it can handle large files
 *        limited only by the available disk capacity for the new .bin outputs.
 *
 * @date 2025-11-29
 */

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char *kUsage =
    "Usage:\n"
    "  data_converter <input_path> <output_path> <mode> [dimension]\n"
    "    mode: 'fvecs' for Faiss-style files\n"
    "          'double' for raw double arrays (requires dimension)\n";

void WriteMetadata(std::ofstream &ofs, uint32_t num_vectors, uint32_t dimension) {
    ofs.write(reinterpret_cast<const char *>(&num_vectors), sizeof(uint32_t));
    ofs.write(reinterpret_cast<const char *>(&dimension), sizeof(uint32_t));
}

void ConvertFvecs(const std::string &input_path, const std::string &output_path) {
    std::ifstream ifs(input_path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Unable to open input file: " + input_path);
    }

    std::ofstream ofs(output_path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        throw std::runtime_error("Unable to open output file: " + output_path);
    }
    WriteMetadata(ofs, 0, 0);  // placeholder, overwritten later

    uint32_t vector_count = 0;
    uint32_t dim = 0;
    std::vector<float> buffer;

    while (true) {
        int32_t current_dim = 0;
        if (!ifs.read(reinterpret_cast<char *>(&current_dim), sizeof(int32_t))) {
            break;  // reached EOF cleanly
        }
        if (current_dim <= 0) {
            throw std::runtime_error("Invalid dimension " + std::to_string(current_dim) +
                                     " in " + input_path);
        }

        if (vector_count == 0) {
            dim = static_cast<uint32_t>(current_dim);
            buffer.resize(dim);
        } else if (static_cast<uint32_t>(current_dim) != dim) {
            throw std::runtime_error("Mixed dimensions detected in " + input_path);
        }

        if (!ifs.read(reinterpret_cast<char *>(buffer.data()),
                      static_cast<std::streamsize>(dim * sizeof(float)))) {
            std::cerr << "Warning: truncated vector encountered after "
                      << vector_count << " complete entries in " << input_path << "."
                      << std::endl;
            break;
        }

        ofs.write(reinterpret_cast<const char *>(buffer.data()),
                  static_cast<std::streamsize>(dim * sizeof(float)));
        ++vector_count;

        if (vector_count == std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("Vector count exceeds uint32_t limit in " + input_path);
        }
    }

    if (vector_count == 0) {
        throw std::runtime_error("No vectors found in " + input_path);
    }

    ofs.seekp(0, std::ios::beg);
    WriteMetadata(ofs, vector_count, dim);

    std::cout << "Converted " << vector_count << " vectors from " << input_path << " to "
              << output_path << " (dimension=" << dim << ")." << std::endl;
}

void ConvertDouble(const std::string &input_path, const std::string &output_path,
                   uint32_t dimension) {
    if (dimension == 0) {
        throw std::runtime_error("Dimension must be positive for double mode");
    }

    std::ifstream ifs(input_path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Unable to open input file: " + input_path);
    }
    std::ofstream ofs(output_path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        throw std::runtime_error("Unable to open output file: " + output_path);
    }
    WriteMetadata(ofs, 0, 0);  // placeholder

    std::vector<double> double_buffer(dimension);
    std::vector<float> float_buffer(dimension);

    uint32_t vector_count = 0;
    while (true) {
        if (!ifs.read(reinterpret_cast<char *>(double_buffer.data()),
                      static_cast<std::streamsize>(dimension * sizeof(double)))) {
            break;
        }
        for (uint32_t d = 0; d < dimension; ++d) {
            float_buffer[d] = static_cast<float>(double_buffer[d]);
        }
        ofs.write(reinterpret_cast<const char *>(float_buffer.data()),
                  static_cast<std::streamsize>(dimension * sizeof(float)));
        ++vector_count;
        if (vector_count == std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("Vector count exceeds uint32_t limit in " + input_path);
        }
    }

    if (vector_count == 0) {
        throw std::runtime_error("No vectors detected in " + input_path);
    }

    ofs.seekp(0, std::ios::beg);
    WriteMetadata(ofs, vector_count, dimension);

    std::cout << "Converted " << vector_count << " vectors from " << input_path << " to "
              << output_path << " (dimension=" << dimension << ")." << std::endl;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << kUsage;
        return EXIT_FAILURE;
    }

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];
    const std::string mode = argv[3];

    try {
        if (mode == "fvecs") {
            ConvertFvecs(input_path, output_path);
        } else if (mode == "double") {
            if (argc < 5) {
                throw std::runtime_error("Missing dimension argument for double mode.");
            }
            const uint32_t dimension = static_cast<uint32_t>(std::stoul(argv[4]));
            ConvertDouble(input_path, output_path, dimension);
        } else {
            throw std::runtime_error("Unknown mode: " + mode);
        }
    } catch (const std::exception &ex) {
        std::cerr << "Conversion failed: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

