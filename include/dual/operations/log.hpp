#pragma once

#include <cmath>

#include "dual/operations/base.hpp"

namespace b2o::dual {
struct log : unary_operation<log> {
  template <class T>
  auto value(const T& v) const {
    return std::log(v);
  }
  template <class T>
  auto dvalue(const duo<T>& n) const {
    assert(n.v > T{0});
    return n.d / n.v;
  }
};
}  // namespace b2o::dual

namespace std {
template <class T, b2o::dual::log::enable_t<T> = 0>
inline auto log(const T& n) {
  return std::invoke(b2o::dual::log{}, n);
}
}  // namespace std
