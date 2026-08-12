#pragma once

#include <cmath>
#include <functional>

#include "dual/operations/base.hpp"

namespace b2o::dual {

struct abs : unary_operation<abs> {
  template <class T>
  auto value(const T& v) const {
    return std::abs(v);
  }

  template <class T>
  auto dvalue(const duo<T>& n) const {
    return n.v < T{0} ? -n.d : (n.v > T{0} ? n.d : T{0});
  }
};

}  // namespace b2o::dual

namespace std {
template <class T, b2o::dual::abs::enable_t<T> = 0>
inline auto abs(const T& n) {
  return std::invoke(b2o::dual::abs{}, n);
}
}  // namespace std
