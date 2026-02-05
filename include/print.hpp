#pragma once
#include <iostream>

const auto print_number =  //
    [](const auto& name, const auto& number) {
      std::cout << name << " = " << number << std::endl;
    };

const auto print_vector =  //
    [](const auto& name, const auto& vector) {
      std::cout << name << " = " << "[";
      for (auto val : vector)
        std::cout << " " << val;
      std::cout << " ]" << std::endl;
    };

const auto print_matrix =  //
    [](const auto& name, const auto& matrix) {
      std::cout << name << " = " << "[";
      for (auto vector : matrix) {
        std::cout << std::endl;
        for (auto val : vector)
          std::cout << " " << val;
      }
      std::cout << std::endl << "]" << std::endl;
    };
