#include <array>

#include "builder.hpp"
#include "helpers/print.hpp"

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

int main() {
  auto optimizer =  //
      b2o::make_optimizer<2>()
          .kernel_radial(2.0)
          .domain_bounds(
              std::array{
                  std::pair{-5.0, 15.0},  //
                  std::pair{-5.0, 15.0}},
              std::array{0.0, 0.0})
          .objective(branin{})
          .build();

  // Warmup phas e
  optimizer.warmup(30);

  // Optimization loop
  for (int iter = 0; iter < 50; ++iter) {
    optimizer.run(1, {100, 0.01, 0.001});
    auto [_, best] = optimizer.best();
    b2o::print_number("iter", iter);
    b2o::print_number("best", best);
  }

  // Report best result
  auto [x_best, y_best] = optimizer.best();
  b2o::print_vector("x_best", x_best);
  b2o::print_number("y_best", y_best);
}
