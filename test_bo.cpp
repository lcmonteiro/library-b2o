#include <iostream>

#include "acquisition/expected_improvement.hpp"
#include "domain/bounds.hpp"
#include "gaussian/process.hpp"
#include "kernel/radial.hpp"
#include "optimization/bayesian.hpp"
#include "optimization/gradient.hpp"

struct branin {
  template <class V>
  auto operator()(const V& x) const {
    double a = 1;
    double b = 5.1 / (4 * M_PI * M_PI);
    double c = 5 / M_PI;
    double r = 6;
    double s = 10;
    double t = 1 / (8 * M_PI);

    double x1 = x[0];
    double x2 = x[1];

    return a * std::pow(x2 - b * x1 * x1 + c * x1 - r, 2) +
           s * (1 - t) * std::cos(x1) + s;
  }
};

using namespace b2o;

int main() {
  auto optimizer = optimization::bayesian<
      acquisition::expected_improvement,
      optimization::gradient,
      gaussian::process<kernel::radial<double>, double, 2>,
      domain::bounds<double, 2>,
      branin>{
      gaussian::make_process<2>(kernel::radial{2.0}),
      domain::bounds{
          std::array{
              std::pair{-5.0, 0.0},  //
              std::pair{10.0, 15.0}},
          std::array{0.0, 0.0}},
      branin{}};

  optimizer.warmup(100);
  for (int iter = 0; iter < 100; ++iter) {
    optimizer.run(1, {0.01, 200});
    auto [x, y] = optimizer.best();
    std::cout << iter << " best = " << y << "\n";
  }

  auto [x_best, y_best] = optimizer.best();
  std::cout << " x=" << x_best[0] << "," << x_best[1]
            << " y=" << y_best << "\n";
}
