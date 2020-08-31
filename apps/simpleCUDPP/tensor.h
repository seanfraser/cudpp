/**
 * Copyright (c) 2020 MIT License by Helen Xu, Sean Fraser
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 **/
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <random>
#include <type_traits>
#include <iostream>
#include <vector>

using namespace std;

namespace exsum_tensor {

// Based on the 4th Edition C++ Programming Language Book
class Rand_int {
 public:
  Rand_int(int lo, int hi, unsigned int seed = 5489u)
      : re(seed), dist(lo, hi) {}
  int operator()() { return dist(re); }

 private:
  mt19937 re;
  uniform_int_distribution<> dist;
};

// float or double
template <typename T>
class Rand_real {
 public:
  Rand_real(T lo, T hi, unsigned int seed = 5489u) : re(seed), dist(lo, hi) {}
  T operator()() { return dist(re); }

 private:
  mt19937 re;
  uniform_real_distribution<T> dist;
};

// float or double
template <typename T>
class Rand_real_exp {
 public:
  Rand_real_exp(T lambda, unsigned int seed = 5489u) : re(seed), dist(lambda) {}
  T operator()() { return dist(re); }

 private:
  mt19937 re;
  exponential_distribution<T> dist;
};

// float or double
template <typename T>
class Rand_real_normal {
 public:
  Rand_real_normal(T mu, T sigma, unsigned int seed = 5489u) : re(seed), dist(mu, sigma) {}
  T operator()() { return dist(re); }

 private:
  mt19937 re;
  normal_distribution<T> dist;
};

// type can be either int, float, or double. N is the number of dimensions
template <typename T, size_t N>
class Tensor {
 public:
  static constexpr size_t order = N;

  Tensor() = default;
  Tensor(const Tensor&) = default;  // copy ctor
  Tensor& operator=(const Tensor&) = default;
  ~Tensor() = default;

  Tensor(const vector<size_t>& side_lens) {
    dim_lens = side_lens;
    size_t s = side_lens[0];
    for (size_t n = 1; n < N; ++n) {
      s *= dim_lens[n];
    }
    elems = vector<T>(s);
  }

  // 0: uniform(hi, lo), 1: exp(lambda), 2: normal(mu, sigma)
  void RandFill(T lo, T hi, int distr_flag = 0) {
    if (is_same<T, int>::value) {
      Rand_int ri{static_cast<int>(lo), static_cast<int>(hi)};
      for (int i = 0; i < size(); ++i) {
        elems[i] = ri();
      }
    } else {
      if (distr_flag == 0) {
        Rand_real<T> rr{static_cast<T>(lo), static_cast<T>(hi)};
        for (int i = 0; i < size(); ++i) {
          elems[i] = rr();
        }
      } else if (distr_flag == 1) {
        Rand_real_exp<T> rr{static_cast<T>(lo)};
        for (int i = 0; i < size(); ++i) {
          elems[i] = rr();
        }
      } else if (distr_flag == 2) {
        Rand_real_normal<T> rr{static_cast<T>(lo), static_cast<T>(hi)};
        for (int i = 0; i < size(); ++i) {
          elems[i] = rr();
        }
      }
    }
  }

  void ZeroFill() {
    for (int i = 0; i < size(); ++i) {
      elems[i] = 0;
    }
  }

  T& GetElt(const vector<size_t>& indices) {
    // convert n-dimensional indices to one dimension - row major
    size_t index = indices[0];
    for (size_t d = 1; d < N; ++d) {
      index = index * dim_lens[d] + indices[d];
    }
    return elems[index];
  }

  size_t getAddress(const vector<size_t>& indices) {
    // convert n-dimensional indices to one dimension - row major
    size_t index = indices[0];
    for (size_t d = 1; d < N; ++d) {
      index = index * dim_lens[d] + indices[d];
    }
    return index;
  }

  const vector<size_t> getMaxIndex() {
    vector<size_t> max_index = dim_lens;
    for (auto& element: max_index)
      element -= 1;
    return max_index;
  }

  const vector<size_t> getMinIndex() {
    vector<size_t> min_index;
    std::fill(min_index.begin(), min_index.end(), 0);
    return min_index;
  }

  void SetElt(const vector<size_t>& indices, T val) {
    // convert n-dimensional indices to one dimension - row major
    int index = indices[0];
    for (size_t d = 1; d < N; ++d) {
      index = index * dim_lens[d] + indices[d];
    }
    elems[index] = val;
  }

  void AppendElt(const vector<size_t>& indices, T val) {
    // convert n-dimensional indices to one dimension - row major
    int index = indices[0];
    for (size_t d = 1; d < N; ++d) {
      index = index * dim_lens[d] + indices[d];
    }
    elems[index] += val;
  }

  // prints a 1D or 2D array to an output stream
  // prints to cout by default
  void Print(vector<size_t> indices = vector<size_t>(N, 0), size_t curr_dim = 1,
             ostream& o = cout) {
    // base case
    if (curr_dim == N) {
      size_t dim_len = dim_lens[curr_dim - 1];
      vector<size_t> index = indices;
      for (size_t i = 0; i < dim_len; ++i) {
        index[curr_dim - 1] = i;
        o << GetElt(index) << " ";
      }
      o << endl;
      return;
    }
    size_t dim_len = dim_lens[curr_dim - 1];
    vector<size_t> index = indices;
    for (size_t i = 0; i < dim_len; ++i) {
      index[curr_dim - 1] = i;
      Print(index, curr_dim + 1);
    }
    o << endl;
  }

  size_t size() const { return elems.size(); }

  T* data() { return elems.data(); }  // C array style access for testing
  const T* data() const { return elems.data(); }

 protected:
  vector<T> elems;  // the tensor data itself (unpadded)
  vector<size_t> dim_lens;
};

} // namespace exsum_tensor

