#pragma once
#include "dual/operations/erf.hpp"
#include "dual/operations/exp.hpp"

namespace b2o::gaussian {

template <class Number>
struct distribution {
  template <class NumberLike>
  auto pdf(const NumberLike& x) const -> NumberLike {
    return inv_sqrt_2pi * std::exp(-half * x * x);
  }

  template <class NumberLike>
  auto cdf(const NumberLike& x) const -> NumberLike {
    return half * (unit + std::erf(x * inv_sqrt_2));
  }

 private:
  static constexpr auto unit = Number{1.0L};
  static constexpr auto half = Number{0.5L};
  static constexpr auto pi =
      Number{3.14159265358979323846264338327950288L};
  static constexpr auto inv_sqrt_2 =
      Number{0.707106781186547524400844362104849L};
  static constexpr auto inv_sqrt_2pi =
      Number{0.39894228040143267793994605993438L};
};

}  // namespace b2o::gaussian
