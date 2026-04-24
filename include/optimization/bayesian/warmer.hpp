#pragma once

#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

#include "domain/normalize.hpp"
#include "optimization/bayesian/runner.hpp"

namespace b2o::optimization::bayesian {

template <
    template <class, class> class Acquisition,  //
    template <class, class> class Optimizer,    //
    class Model, class Functor, class Domain>
class warmer {
 public:
  using model_t = Model;
  using domain_x_t = Domain;
  using functor_t = Functor;

  using number_t = typename model_t::number_t;
  using sample_t = typename model_t::sample_t;
  using samples_t = std::vector<sample_t>;

  using builder_y_t = domain::normalize_builder<number_t>;
  using domain_y_t = typename builder_y_t::normalize;

  using runner_t = bayesian::runner<
      Acquisition, Optimizer,  //
      model_t, functor_t, domain_x_t, domain_y_t>;

  warmer(
      model_t model, domain_x_t domain, functor_t functor)
      : model_(std::move(model)),
        functor_(std::move(functor)),
        domain_x_(std::move(domain)) {
  }

  [[nodiscard]]
  auto warmup(std::size_t steps) -> runner_t {
    assert(steps > 0);

    model_t model{model_};
    samples_t samples{};
    builder_y_t builder_y{};
    samples.reserve(steps);

    // Initial sample
    auto best_x = domain_x_.start();
    auto best_y = functor_(best_x);

    samples.emplace_back(best_x, best_y);
    builder_y.emplace(best_y);

    // Random exploration
    for (std::size_t i = 1; i < steps; ++i) {
      const auto next_x = domain_x_.random();
      const auto next_y = functor_(next_x);

      samples.emplace_back(next_x, next_y);
      builder_y.emplace(next_y);

      if (next_y < best_y) {
        best_x = next_x;
        best_y = next_y;
      }
    }

    // Build normalization domain
    const auto domain_y = builder_y.build();

    // Fill model
    for (std::size_t i = 0; i < steps; ++i) {
      const auto& [next_x, next_y] = samples[i];
      model.emplace(next_x, domain_y.project(next_y));
    }

    return runner_t{
        std::move(model),
        functor_,
        domain_x_,
        domain_y,
        {best_x, domain_y.project(best_y)}};
  }

 private:
  model_t model_;
  functor_t functor_;
  domain_x_t domain_x_;
};

}  // namespace b2o::optimization::bayesian
