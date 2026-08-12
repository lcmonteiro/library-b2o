#pragma once

#include <cmath>
#include <functional>

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

struct log1p : unary_operation<log1p> {
  template <class T>
  auto value(const T& v) const {
    return std::log1p(v);
  }
  template <class T>
  auto dvalue(const duo<T>& n) const {
    assert(n.v > T{-1});
    return n.d / (T{1} + n.v);
  }
};
}  // namespace b2o::dual

namespace std {
template <class T, b2o::dual::log::enable_t<T> = 0>
inline auto log(const T& n) {
  return std::invoke(b2o::dual::log{}, n);
}
template <class T, b2o::dual::log1p::enable_t<T> = 0>
inline auto log1p(const T& n) {
  return std::invoke(b2o::dual::log1p{}, n);
}
}  // namespace std
