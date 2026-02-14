#pragma once

#include <utility>

#include "dual/number.hpp"
#include "helpers/functional.hpp"

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
  template <class Input>
  auto minimize(Input&& x) const {
    // Lambda to perform a single gradient step
    const auto step = [this](auto& n, auto dv) {
      n -= config_.rate * dv;
    };
    return optimize(std::forward<Input>(x), step);
  }

  /// @brief Maximize objective starting from x
  /// @param x Initial point
  /// @return Optimized point
  template <class Input>
  auto maximize(Input&& x) const {
    // Lambda to perform a single gradient step
    const auto step = [this](auto& n, auto dv) {
      n += config_.rate * dv;
    };
    return optimize(std::forward<Input>(x), step);
  }

 protected:
  template <class Input, class Step>
  auto optimize(Input x, Step step) const -> Input {
    // Lambda to check convergence
    const auto done = [this](auto dv) {
      return std::abs(dv) < config_.eps;
    };
    // Lambda to reseed dual numbers
    const auto seed = [](auto& dn, auto vn) {
      dn.value(vn);
    };
    print_vector("init", x);
    auto dinput = dual::make_array(x);
    for (std::size_t s = 0; s < config_.steps; ++s) {
      const auto dresult = functor_(dinput);
      each(step, x, dresult.dvalue());
      print_number("iter", s);
      print_number("objective", dresult.value());
      print_vector("gradient", dresult.dvalue());
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
