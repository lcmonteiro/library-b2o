#pragma once

#include <cassert>
#include <cstddef>

namespace b2o {

///@brief Apply a function elementwise to multiple
/// containers
///@tparam F Callable type
///@tparam T Head container type
///@tparam Ts Tail container types
///@param doit Function to apply
///@param head First container
///@param tail Other containers (sizes must match first)
///@note All containers must have the same size as 'first'
template <class F, class T, class... Ts>
inline void each(F doit, T&& head, Ts&&... tail) {
  for (std::size_t i = 0; i < head.size(); ++i) {
    doit(head[i], tail[i]...);
  }
}

///@brief Check if a predicate returns true for all elements
/// of containers
///@tparam F Callable type
///@tparam T Head container type
///@tparam Ts Tail container types
///@param check Predicate to check
///@param head First container
///@param tail Other containers (sizes must match first)
///@return True if predicate is true for all elements, false
/// otherwise
///@note All containers must have the same size as 'first'
template <class F, class T, class... Ts>
inline bool all(F check, const T& head, const Ts&... tail) {
  for (std::size_t i = 0; i < head.size(); ++i) {
    if (!check(head[i], tail[i]...))
      return false;
  }
  return true;
}

}  // namespace b2o
