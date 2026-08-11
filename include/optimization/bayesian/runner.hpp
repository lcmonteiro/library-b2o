#pragma once

#include <cstddef>
#include <utility>

namespace b2o::optimization::bayesian {

template <
    template <class, class> class Acquisition,
    template <class, class> class Optimizer, class Model,
    class Functor, class DomainX, class DomainY>
class runner {
 public:
  using model_t = Model;
  using domain_x_t = DomainX;
  using domain_y_t = DomainY;
  using functor_t = Functor;

  using number_t = typename model_t::number_t;
  using sample_t = typename model_t::sample_t;

  using acquisition_t = Acquisition<model_t, number_t>;
  using optimizer_t = Optimizer<acquisition_t, number_t>;
  using config_t = typename optimizer_t::config_t;

  runner(
      model_t model,        //
      functor_t functor,    //
      domain_x_t domain_x,  //
      domain_y_t domain_y,  //
      sample_t best)
      : model_(std::move(model)),
        functor_(std::move(functor)),
        domain_x_(std::move(domain_x)),
        domain_y_(std::move(domain_y)),
        best_(std::move(best)) {
  }

  [[nodiscard]]
  auto best() const -> const sample_t {
    auto& [best_x, best_y] = best_;
    return {best_x, domain_y_.inverse(best_y)};
  }

  void run(std::size_t steps, const config_t& config) {
    auto& [best_x, best_y] = best_;

    for (std::size_t s = 0; s < steps; ++s) {
      acquisition_t acq{model_, best_y};
      optimizer_t opt{acq, config};

      const auto next_x = propose(acq, opt, best_x);

      const auto real_y = functor_(next_x);
      const auto next_y = domain_y_.project(real_y);

      model_.emplace(next_x, next_y);
      if (next_y < best_y) {
        best_x = next_x;
        best_y = next_y;
      }
    }
  }

 protected:
  // Maximize the acquisition function from several starting
  // points (one exploiting the current best, the rest
  // exploring the domain at random) and keep whichever local
  // optimum scores highest, to avoid getting stuck in the
  // first local maximum found near the incumbent.
  static constexpr std::size_t kRestarts = 8;

  template <class AcqFn, class OptFn, class InputX>
  auto propose(
      const AcqFn& acq, OptFn& opt, const InputX& best_x)
      -> InputX {
    auto next_x = domain_x_.project(
        opt.maximize(domain_x_.generate(best_x)));
    auto next_score = acq(next_x);

    for (std::size_t r = 1; r < kRestarts; ++r) {
      const auto candidate_x = domain_x_.project(
          opt.maximize(domain_x_.random()));
      const auto candidate_score = acq(candidate_x);
      if (next_score < candidate_score) {
        next_x = candidate_x;
        next_score = candidate_score;
      }
    }
    return next_x;
  }

 private:
  model_t model_;
  functor_t functor_;
  domain_x_t domain_x_;
  domain_y_t domain_y_;
  sample_t best_;
};

}  // namespace b2o::optimization::bayesian
