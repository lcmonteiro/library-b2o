#pragma once

#include <cmath>
#include <cstddef>
#include <utility>

#include "dual/number.hpp"
#include "helpers/functional.hpp"
#include "helpers/print.hpp"

namespace b2o::optimization {

/// @brief Configuration for gradient descent optimizer
template <class Number>
struct gradient_config {
  using number_t = Number;

  std::size_t steps{100};  ///< Maximum number of iterations
  number_t rate{1e-2};     ///< Learning rate
  number_t eps{1e-6};      ///< Minimum step size
};

template <class Number>
gradient_config(std::size_t, Number, Number)
    -> gradient_config<Number>;

/// @brief Default projection (unconstrained)
struct identity {
  template <class Input>
  auto operator()(Input&& x) const -> Input&& {
    return std::forward<Input>(x);
  }
};

/// @brief Gradient descent optimizer using forward-mode
/// autodiff
/// @tparam Functor Objective function type
/// @tparam Number Numeric type
template <class Functor, class Number>
class gradient {
 public:
  using number_t = Number;
  using config_t = gradient_config<Number>;

  /// @brief Construct gradient optimizer
  /// @param functor Objective function
  /// @param config Gradient configuration
  template <class Fn>
  explicit gradient(Fn&& functor, const config_t& config)
      : functor_{std::forward<Fn>(functor)},
        config_{config} {
  }

  /// @brief Minimize objective starting from x
  /// @param x Initial point
  /// @return Optimized point
  template <class Input, class Project = identity>
  auto minimize(Input&& x, Project project = {}) const {
    // Lambda to perform a single gradient step
    const auto step = [](auto& n, auto dv, auto rate) {
      n -= rate * dv;
    };
    return optimize(std::forward<Input>(x), step, project);
  }

  /// @brief Maximize objective starting from x
  /// @param x Initial point
  /// @return Optimized point
  template <class Input, class Project = identity>
  auto maximize(Input&& x, Project project = {}) const {
    // Lambda to perform a single gradient step
    const auto step = [](auto& n, auto dv, auto rate) {
      n += rate * dv;
    };
    return optimize(std::forward<Input>(x), step, project);
  }

 protected:
  /// @brief Step size with the gradient norm clipped
  /// to one, so steep regions can not throw the iterate
  /// far away from the current point.
  template <class Gradient>
  auto clipped_rate(const Gradient& dv) const -> number_t {
    auto norm = number_t{0};
    for (const auto& d : dv)
      norm += d * d;
    norm = std::sqrt(norm);
    return norm > number_t{1} ? config_.rate / norm
                              : config_.rate;
  }

  template <class Input, class Step, class Project>
  auto optimize(Input x, Step step, Project project) const
      -> Input {
    // Lambda to check convergence
    const auto done = [this](auto dv) {
      return std::abs(dv) < config_.eps;
    };
    // Lambda to reseed dual numbers
    const auto seed = [](auto& dn, auto vn) {
      dn.value(vn);
    };
    // print_vector("init", x);
    auto dinput = dual::make_array(x);
    for (std::size_t s = 0; s < config_.steps; ++s) {
      // print_number("iter", s);
      const auto dresult = functor_(dinput);
      const auto rate = clipped_rate(dresult.dvalue());
      each(
          [&](auto& n, auto dv) { step(n, dv, rate); },
          x,
          dresult.dvalue());
      x = project(x);
      // print_number("objective", dresult.value());
      // print_vector("gradient", dresult.dvalue());
      // print_vector("input", x);
      if (all(done, dresult.dvalue()))
        break;
      each(seed, dinput, x);
    }
    return x;
  }

 private:
  Functor functor_;  ///< Objective function
  config_t config_;  ///< Gradient configuration
};

template <class Functor, class Number>
gradient(Functor, const gradient_config<Number>&)
    -> gradient<Functor, Number>;

}  // namespace b2o::optimization
