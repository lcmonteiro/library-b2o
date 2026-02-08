#pragma once

#include <iostream>

#include "dual/number.hpp"

// ============================================================
// operator<< for dual::number
// prints value and derivative values
// ============================================================
namespace b2o {

template <class T>
auto& operator<<(
    std::ostream& os, const dual::number<T>& n) {
  os << n.value() << " [";
  for (std::size_t i = 0; i < n.size(); ++i) {
    if (i > 0)
      os << ", ";
    os << n.dvalue(i);
  }
  os << "]";
  return os;
}

// ============================================================
// Print Helpers
// ============================================================
const auto print_number =  //
    [](const auto& name, const auto& number) {
      std::cout << name << " = " << number << std::endl;
    };

const auto print_vector =  //
    [](const auto& name, const auto& vector) {
      std::cout << name << " = [";
      for (auto&& val : vector)
        std::cout << " " << val;
      std::cout << " ]" << std::endl;
    };

const auto print_matrix =  //
    [](const auto& name, const auto& matrix) {
      std::cout << name << " = [";
      for (auto&& row : matrix) {
        std::cout << std::endl << "  [";
        for (auto&& val : row)
          std::cout << " " << val;
        std::cout << " ]";
      }
      std::cout << std::endl << "]" << std::endl;
    };
}  // namespace b2o
