#pragma once

#include <cmath>

#include "dual/operations/base.hpp"

namespace b2o::dual {
struct sqrt : unary_operation<sqrt> {
  template <class T>
  auto value(const T& v) const {
    return std::sqrt(v);
  }
  template <class T>
  auto dvalue(const duo<T>& n) const {
    return n.d / (T{2} * std::sqrt(n.v));
  }
};
}  // namespace b2o::dual

namespace std {
template <class T, b2o::dual::sqrt::enable_t<T> = 0>
inline auto sqrt(const T& n) {
  return std::invoke(b2o::dual::sqrt{}, n);
}
}  // namespace std
