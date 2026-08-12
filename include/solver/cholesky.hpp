#pragma once

#include <type_traits>

#include "dual/operations.hpp"

namespace b2o::math {

template <class Number>
class cholesky {
 public:
  template <class MatrixA, class MatrixL>
  auto build(
      const MatrixA& a,  //
      MatrixL& l,        //
      size_t beg = 0) const -> void {
    l.resize(beg);
    for (size_t i = beg; i < a.size(); ++i) {
      l.emplace_back(i + 1);
      for (size_t j = 0; j < i; ++j) {
        const auto sum = accumulate{0, j};
        l[i][j] = sum(l[i], l[j], a[i][j]) / l[j][j];
      }
      const auto sum = accumulate{0, i};
      l[i][i] = std::sqrt(sum(l[i], l[i], a[i][i]));
    }
  }

  template <class MatrixL, class VectorB, class VectorY>
  auto forward(
      const MatrixL& l,  //
      const VectorB& b,  //
      VectorY& y,        //
      size_t beg = 0) const -> void {
    const auto n = b.size();
    y.resize(n);
    for (size_t i = beg; i < n; ++i) {
      const auto sum = accumulate{0, i};
      y[i] = sum(l[i], y, b[i]) / l[i][i];
    }
  }

  template <class MatrixL, class VectorY, class VectorX>
  auto backward(
      const MatrixL& l,  //
      const VectorY& y,  //
      VectorX& x) const -> void {
    const auto n = y.size();
    x.resize(n);
    for (size_t i = n; i-- > 0;) {
      const auto col = column{l, i};
      const auto sum = accumulate{i + 1, n};
      x[i] = sum(col, x, y[i]) / l[i][i];
    }
  }

 protected:
  struct accumulate {
    const size_t beg;
    const size_t end;
    template <
        class VectorA,  //
        class VectorB,  //
        class NumberLike>
    auto operator()(
        const VectorA& a,  //
        const VectorB& b,  //
        const NumberLike& init) const {
      // Accumulate the a[k]*b[k] products on their own
      // (they share one type) before combining with init,
      // instead of folding into a variable fixed to init's
      // type: init and the products can be different types
      // (e.g. one dual, one plain) when a/b and init come
      // from different call sites, and init's type may not
      // be assignable from that combination.
      using term_t =
          std::decay_t<decltype(a[beg] * b[beg])>;
      auto sum = term_t{};
      for (size_t k = beg; k < end; ++k) {
        sum = sum + a[k] * b[k];
      }
      return init - sum;
    }
  };

  template <class Matrix>
  struct column {
    const Matrix& data;
    const size_t index;
    const auto& operator[](size_t i) const {
      return data[i][index];
    }
  };
  template <class Matrix>
  column(const Matrix&, size_t) -> column<Matrix>;
};

}  // namespace b2o::math
