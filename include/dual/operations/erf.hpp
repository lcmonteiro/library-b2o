#pragma once
#include <cmath>

#include "dual/operations/base.hpp"

namespace b2o::dual {
struct erf : unary_operation<erf> {
  template <class T>
  auto value(const T& v) const {
    return std::erf(v);
  }
  template <class T>
  auto dvalue(const duo<T>& n) const {
    return two_over_sqrt_pi<T> * std::exp(-n.v * n.v) * n.d;
  }

 private:
  template <class T>
  static constexpr T two_over_sqrt_pi =
      T{1.12837916709551257389615890312154517L};
};
}  // namespace b2o::dual

namespace std {
template <class T, b2o::dual::erf::enable_t<T> = 0>
inline auto erf(const T& n) {
  return std::invoke(b2o::dual::erf{}, n);
}
}  // namespace std
