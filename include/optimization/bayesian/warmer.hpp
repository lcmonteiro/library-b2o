#pragma once

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include "domain/normalize.hpp"
#include "gaussian/fit.hpp"
#include "kernel/radial.hpp"
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
    if constexpr (std::is_same_v<
                      typename model_t::kernel_t,
                      kernel::radial<number_t>>) {
      // Radial kernels have a single lengthscale
      // hyperparameter: fit it to the warmup data by
      // maximizing the GP log marginal likelihood instead
      // of trusting the user-supplied value as-is (it's
      // only used as the search's starting guess). The
      // kernel is immutable once built, so the fitted model
      // is constructed fresh rather than mutated in place.
      auto fit_samples = samples_t{};
      fit_samples.reserve(steps);
      for (std::size_t i = 0; i < steps; ++i) {
        const auto& [next_x, next_y] = samples[i];
        fit_samples.emplace_back(
            next_x, domain_y.project(next_y));
      }
      const auto noise = model_.noise();
      const auto sigma = gaussian::fit_radial_lengthscale(
          fit_samples, noise, model_.kernel().sigma());

      return runner_t{
          model_t{
              kernel::radial<number_t>{sigma}, fit_samples,
              noise},
          functor_,
          domain_x_,
          domain_y,
          {best_x, domain_y.project(best_y)}};
    } else {
      model_t model{model_};
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
  }

 private:
  model_t model_;
  functor_t functor_;
  domain_x_t domain_x_;
};

}  // namespace b2o::optimization::bayesian
