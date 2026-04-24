#pragma once
#include "dual/operations/erf.hpp"
#include "dual/operations/exp.hpp"
#include "dual/operations/log.hpp"

namespace b2o::gaussian {

template <class Number>
struct distribution {
  template <class NumberLike>
  auto pdf(const NumberLike& x) const -> NumberLike {
    return inv_sqrt_2pi * std::exp(-half * x * x);
  }
  template <class NumberLike>
  auto log_pdf(const NumberLike& x) const -> NumberLike {
    return -half * x * x - log_sqrt_2pi;
  }

  template <class NumberLike>
  auto cdf(const NumberLike& x) const -> NumberLike {
    return half * (unit + std::erf(x * inv_sqrt_2));
  }
  template <class NumberLike>
  auto log_cdf(const NumberLike& x) const -> NumberLike {
    return std::log(cdf(x));
  }
  template <class NumberLike>
  auto log_cdf_(const NumberLike& x) const -> NumberLike {
    constexpr auto a = Number{1.702};
    constexpr auto b = Number{0.044715};
    const auto t = a * x + b * x * x * x;
    if (NumberLike{0} < t) {
      return -std::log1p(std::exp(-t));
    } else {
      return t - std::log1p(std::exp(t));
    }
  }

 private:
  static constexpr auto unit = Number{1.0L};
  static constexpr auto half = Number{0.5L};
  static constexpr auto inv_sqrt_2 =
      Number{0.70710678118654752440084436210485L};
  static constexpr auto inv_sqrt_2pi =
      Number{0.39894228040143267793994605993438L};
  static constexpr auto log_sqrt_2pi =
      Number{0.91893853320467274178032973640562L};
};

}  // namespace b2o::gaussian
