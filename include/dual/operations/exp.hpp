#pragma once

#include <cmath>

#include "dual/operations/base.hpp"

namespace b2o::dual {
struct exp : unary_operation<exp> {
  template <class T>
  auto value(const T& v) const {
    return std::exp(v);
  }
  template <class T>
  auto dvalue(const duo<T>& n) const {
    return std::exp(n.v) * n.d;
  }
};
}  // namespace b2o::dual

namespace std {
template <class T, b2o::dual::exp::enable_t<T> = 0>
inline auto exp(const T& n) {
  return std::invoke(b2o::dual::exp{}, n);
}
}  // namespace std
