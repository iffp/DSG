/**
 * @file flat_vectors.h
 * @brief Contiguous storage wrapper for a collection of fixed-length vectors.
 *
 * @tparam T Scalar element type. Defaults to float, but can be instantiated
 *         with integral types for lightweight tensor-style integer storage.
 *
 * The class stores all values inside a single std::vector<T> buffer while
 * exposing light-weight row views so callers can access rows via familiar
 * operator[] syntax without paying the cost of std::vector<std::vector<T>>.
 */
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

template <typename T = float>
class FlatVectors {
public:
    FlatVectors() = default;
    FlatVectors(size_t count, size_t dim) {
        resize(count, dim);
    }

    void clear() {
        storage_.clear();
        count_ = 0;
        dim_ = 0;
    }

    void resize(size_t count, size_t dim) {
        if (dim == 0) {
            throw std::runtime_error("FlatVectors::resize requires a positive dimension.");
        }
        dim_ = dim;
        count_ = count;
        storage_.assign(count * dim, T{});
    }

    void resize(size_t count) {
        ensure_dimension();
        count_ = count;
        storage_.assign(count * dim_, T{});
    }

    void shrink(size_t count) {
        ensure_dimension();
        if (count > count_) {
            throw std::runtime_error("FlatVectors::shrink cannot increase size.");
        }
        count_ = count;
        storage_.resize(count * dim_);
    }

    void set_dimension(size_t dim) {
        if (dim_ != 0 && dim_ != dim) {
            throw std::runtime_error("FlatVectors dimension already set.");
        }
        dim_ = dim;
    }

    size_t size() const {
        return count_;
    }

    size_t dim() const {
        return dim_;
    }

    bool empty() const {
        return count_ == 0;
    }

    T *operator[](size_t idx) {
        return storage_.data() + idx * dim_;
    }

    const T *operator[](size_t idx) const {
        return storage_.data() + idx * dim_;
    }

    T *at(size_t idx) {
        if (idx >= count_) {
            throw std::out_of_range("FlatVectors::at");
        }
        return (*this)[idx];
    }

    const T *at(size_t idx) const {
        if (idx >= count_) {
            throw std::out_of_range("FlatVectors::at");
        }
        return (*this)[idx];
    }

    T *data() {
        return storage_.data();
    }

    const T *data() const {
        return storage_.data();
    }

    T &operator()(size_t row, size_t col) {
        assert(row < count_);
        assert(col < dim_);
        return storage_[row * dim_ + col];
    }

    const T &operator()(size_t row, size_t col) const {
        assert(row < count_);
        assert(col < dim_);
        return storage_[row * dim_ + col];
    }

    void assign_row(size_t row_idx, const T *values) {
        assert(row_idx < count_);
        std::memcpy(storage_.data() + row_idx * dim_, values, dim_ * sizeof(T));
    }

    void push_back(const T *values) {
        ensure_dimension();
        storage_.insert(storage_.end(), values, values + dim_);
        ++count_;
    }

private:
    void ensure_dimension() const {
        if (dim_ == 0) {
            throw std::runtime_error("FlatVectors dimension is zero.");
        }
    }

    std::vector<T> storage_;
    size_t count_{0};
    size_t dim_{0};
};

