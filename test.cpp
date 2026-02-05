#include <functional>
#include <iostream>
#include <utility>
#include <vector>

#include "acquisition/expected_improvement.hpp"
#include "dual/number.hpp"
#include "dual/operations.hpp"
#include "gaussian/process.hpp"
#include "kernel/radial.hpp"
#include "optimization/gradient.hpp"
#include "print.hpp"
#include "solver/cholesky.hpp"

using namespace b2o;

int main(int argc, char* argv[]) {
  auto gradient = optimization::gradient{
      [](auto x) { return x[0] * x[0]; },  //
      optimization::gradient_config{.1, 10}};
  print_vector("m", gradient.minimize(std::array{37.0}));

  auto solver = math::cholesky<double>{};
  auto l = std::vector<std::vector<double>>{};
  solver.build(
      std::vector{
          std::vector{5.0},            //
          std::vector{2.0, 4.0},       //
          std::vector{1.0, 0.5, 3.0},  //
          std::vector{0.5, 1.0, 0.2, 2.0}},
      l);
  print_matrix("l", l);
  auto y = std::vector<double>{};
  solver.forward(l, std::vector{7.0, 8.0, 5.0, 3.0}, y);
  print_vector("y", y);
  auto x = std::vector<double>{};
  solver.backward(l, y, x);
  print_vector("x", x);

  auto gp = gaussian::make_process(
      kernel::radial{1.0},  //
      std::vector{
          std::pair{std::array{1.0}, 1.0},
          std::pair{std::array{2.0}, 3.0},
          std::pair{std::array{3.0}, 5.0}});
  // gp.emplace(std::pair{std::array{1.0}, 1.0});
  // gp.emplace(std::array{2.0}, 3.0);
  // gp.emplace(std::array{3.0}, 5.0);

  for (double x : {1.5, 1.0, 2.0, 3.0}) {
    auto [m, v] = gp.predict(std::array{x});
    std::cout << x << " mean=" << m << " var=" << v << "\n";
  }

  auto ei = acquisition::expected_improvement{gp, 1.3};
  const auto ei_value = ei(dual::make_array<1>(1.6));
  std::cout << "ei = " << ei_value.value() << std::endl;
  print_vector("dv", ei_value.dvalue());
  //
  return 0;
}
