#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

#include "solver/cholesky.hpp"

namespace b2o::gaussian {
// @brief Gaussian Process Regression
// (single test-point prediction))
//
// Given:
//   x*      Test input
//   X       Training inputs  [x1, x2, ..., xn] ,
//   Y       Training targets [y1, y2, ..., yn].T ,
//   k(.,.)  Kernel function ,
//   sn.var   Noise variance ,
//
// Kernel matrix:
//   K = K(X, X) + sn.var * I ,
//
//   where:
//     K_ij = k(xi, xj)     ,
//
// Kernel vector:
//   k_* = k(X, x*) =      ,
//        [ k(x1, x*) ]    ,
//        [ k(x2, x*) ]    ,
//        [     ...   ]    ,
//
// Cholesky decomposition:
//   K = L L.T ,
//
// Intermediate variables used in the implementation:
//
//   z = L.inv * y                (forward substitution)),
//   a = L.T.inv * z = K.inv * y  (backward substitution)),
//
//   v = L.inv * k*               (forward substitution)
//
// Predictive mean ,
//   mean(x*) = k*.T * a ,
//
// Predictive variance:
//   var(x*) = k(x*, x*) − dot(v,v))
//
//
template <class Kernel, class Number, std::size_t Dimension>
class process {
  using Solver = math::cholesky<Number>;
  template <class NumberLike>
  using Vector = std::vector<NumberLike>;
  template <class NumberLike>
  using Matrix = std::vector<Vector<NumberLike>>;
  template <class NumberLike>
  using Input = std::array<NumberLike, Dimension>;
  template <class NumberLike>
  using Inputs = std::vector<Input<NumberLike>>;
  template <class NumberLike>
  using Sample = std::pair<Input<NumberLike>, NumberLike>;
  template <class NumberLike>
  using Samples = std::vector<Sample<NumberLike>>;
  template <class NumberLike>
  using Result = std::pair<Number, Number>;

  static constexpr auto kJitter = 1e-12;

 public:
  using number_t = Number;
  using sample_t = Sample<Number>;

  template <class Dataset = Samples<Number>>
  process(
      const Kernel& kernel,    //
      const Dataset& samples,  //
      const Number noise)
      : k_func_{kernel},
        k_noise_{std::max(noise * noise, kJitter)} {
    samples_init(samples);
    kernel_init();
    solve_full();
  }

  auto size() const -> size_t {
    assert(k_.size() == l_.size());
    return k_.size();
  }

  auto emplace(const Input<Number>& x, const Number& y)
      -> void {
    samples_update(x, y);
    kernel_update(x);
    solve_last();
  }

  auto emplace(const Sample<Number>& sample) -> void {
    const auto& [x, y] = sample;
    emplace(x, y);
  }

  template <class NumberLike>
  auto predict(const Input<NumberLike>& s) const {
    const auto solver = Solver{};
    const auto ss = k_func_(s, s);
    const auto xs = kernel_xs(s);
    auto v = Vector<NumberLike>{};
    solver.forward(l_, xs, v);
    const auto mean = dot_product(xs, a_);
    const auto variance = ss - dot_product(v, v);
    return std::tuple{
        mean, std::max(variance, NumberLike{0})};
  }

 protected:
  template <class Container>
  auto samples_init(const Container& samples) -> void {
    x_.clear();
    y_.clear();
    x_.reserve(samples.size());
    y_.reserve(samples.size());
    for (const auto& [x, y] : samples) {
      x_.emplace_back(x);
      y_.emplace_back(y);
    }
  }

  auto samples_update(
      const Input<Number>& x, const Number& y) -> void {
    x_.emplace_back(x);
    y_.emplace_back(y);
  }

  auto kernel_init() -> void {
    const auto size = x_.size();
    k_.resize(size);
    for (size_t i = 0; i < size; ++i) {
      k_[i].resize(size);
      for (size_t j = 0; j < size; ++j) {
        if (i == j) {
          k_[i][j] = k_func_(x_[i], x_[j]) + k_noise_;
        } else {
          k_[i][j] = k_func_(x_[i], x_[j]);
        }
      }
    }
  }

  auto kernel_update(const Input<Number>& x) -> void {
    const auto size = k_.size();
    for (size_t i = 0; i < size; ++i) {
      k_[i].emplace_back(k_func_(x_[i], x));
    }
    k_.emplace_back(size + 1);
    for (size_t j = 0; j < size; ++j) {
      k_.back()[j] = k_func_(x, x_[j]);
    }
    k_.back()[size] = k_func_(x, x) + k_noise_;
  }

  template <class NumberLike>
  auto kernel_xs(const Input<NumberLike>& s) const
      -> Vector<NumberLike> {
    auto result = std::vector<NumberLike>{};
    result.reserve(x_.size());
    std::transform(
        std::cbegin(x_),  //
        std::cend(x_),    //
        std::back_inserter(result),
        [&s, this](const auto& x) {
          return k_func_(x, s);
        });
    return result;
  }

  template <class NumberLikeA, class NumberLikeB>
  auto dot_product(
      const Vector<NumberLikeA>& a,
      const Vector<NumberLikeB>& b) const -> NumberLikeA {
    return std::inner_product(
        std::cbegin(a),
        std::cend(a),
        std::cbegin(b),
        NumberLikeA{});
  }

  template <class NumberLikeA, class NumberLikeB>
  auto dot_product(
      const Vector<NumberLikeA>& a,
      const Matrix<NumberLikeB>& b) const
      -> Vector<NumberLikeA> {
    auto result = Vector<NumberLikeA>{};
    result.reserve(b.size());
    std::transform(
        std::cbegin(b),  //
        std::cend(b),    //
        std::back_inserter(result),
        [this, &a](const auto& r) {
          return this->dot_product(a, r);
        });
    return result;
  }

  auto solve_full() -> void {
    const auto solver = Solver{};
    solver.build(k_, l_);
    solver.forward(l_, y_, z_);
    solver.backward(l_, z_, a_);
  }

  auto solve_last() -> void {
    const auto solver = Solver{};
    solver.build(k_, l_, l_.size());
    solver.forward(l_, y_, z_, z_.size());
    solver.backward(l_, z_, a_);
  }

 private:
  Kernel k_func_;
  Number k_noise_;
  Matrix<Number> k_;
  Matrix<Number> l_;
  Vector<Number> z_;
  Vector<Number> a_;
  Inputs<Number> x_;
  Vector<Number> y_;
};

template <std::size_t Dimension, class Kernel>
inline auto make_process(const Kernel& kernel) {
  using Number = typename Kernel::number_t;
  return process<Kernel, Number, Dimension>{
      kernel, {}, Number{0.0}};
}

template <std::size_t Dimension, class Kernel, class Number>
inline auto make_process(
    const Kernel& kernel,  //
    const Number& noise) {
  return process<Kernel, Number, Dimension>{
      kernel, {}, noise};
}

template <class Kernel, class Dataset>
inline auto make_process(
    const Kernel& kernel,  //
    const Dataset& dataset) {
  using KNumber = typename Kernel::number_t;
  using DSample = typename Dataset::value_type;
  using DInput = typename DSample::first_type;
  using DNumber = typename DInput::value_type;
  static_assert(std::is_same_v<KNumber, DNumber>);
  constexpr auto Dimension = std::tuple_size_v<DInput>;
  return process<Kernel, DNumber, Dimension>{
      kernel, dataset, DNumber{0}};
}

template <class Kernel, class Dataset, class Number>
inline auto make_process(
    const Kernel& kernel,    //
    const Dataset& dataset,  //
    const Number& noise) {
  using KNumber = typename Kernel::number_t;
  using DSample = typename Dataset::value_type;
  using DInput = typename DSample::first_type;
  using DNumber = typename DInput::value_type;
  static_assert(std::is_same_v<Number, KNumber>);
  static_assert(std::is_same_v<Number, DNumber>);
  constexpr auto Dimension = std::tuple_size_v<DInput>;
  return process<Kernel, Number, Dimension>{
      kernel, dataset, noise};
}

}  // namespace b2o::gaussian
