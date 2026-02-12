#pragma once

#include <cstddef>
#include <utility>

#include "helpers/debug.hpp"

static auto debug = debug_3d{"./bayesian_debug.json"};

namespace b2o::optimization {

template <
    template <class, class> class Acquisition,  //
    template <class, class> class Optimizer,    //
    class Model, class Domain, class Functor>
class bayesian {
 public:
  using number_t = typename Model::number_t;
  using sample_t = typename Model::sample_t;

  using acquisition_t = Acquisition<Model, number_t>;
  using optimizer_t = Optimizer<acquisition_t, number_t>;
  using config_t = typename optimizer_t::config_t;

  bayesian(Model model, Domain domain, Functor functor)
      : model_{std::move(model)},
        domain_{std::move(domain)},
        functor_{std::move(functor)},
        best_{domain_.start(), functor_(domain_.start())} {
    model_.emplace(best_);
    // debug entry
    debug.print(std::pair{best_.first, best_.second}, model_);
  }

  auto best() const -> const sample_t& {
    return best_;
  }

  auto warmup(size_t steps) -> void {
    auto& [x_best, y_best] = best_;

    for (std::size_t s = 0; s < steps; ++s) {
      const auto x_next = domain_.random();
      const auto y_next = functor_(x_next);
      model_.emplace(x_next, y_next);
      if (y_next < y_best) {
        x_best = x_next;
        y_best = y_next;
      }
      // debug entry
      debug.print(std::pair{x_next, y_next}, model_);
    }
  }

  auto run(size_t steps, config_t config) -> void {
    auto& [x_best, y_best] = best_;

    for (std::size_t s = 0; s < steps; ++s) {
      const auto acq = acquisition_t{model_, y_best};
      const auto opt = optimizer_t{acq, config};

      const auto x_gen = domain_.generate(x_best);
      const auto x_max = opt.maximize(x_gen);
      const auto x_next = domain_.project(x_max);
      const auto y_next = functor_(x_next);
      model_.emplace(x_next, y_next);
      if (y_next < y_best) {
        x_best = x_next;
        y_best = y_next;
      }
      // debug entry
      debug.print(std::pair{x_next, y_next}, model_, acq);
    }
  }

 private:
  Model model_;
  Domain domain_;
  Functor functor_;
  sample_t best_;
};

}  // namespace b2o::optimization
