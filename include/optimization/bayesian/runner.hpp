#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

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

  // @param scan_samples Number of cheap (gradient-free)
  //   acquisition evaluations used to find promising starting
  //   points before refining any of them.
  // @param refine_top Number of best-scoring scan candidates
  //   that get refined via gradient ascent; the highest-scoring
  //   refined point is used as the next sample.
  void run(
      std::size_t steps, const config_t& config,
      std::size_t scan_samples = 200,
      std::size_t refine_top = 3) {
    auto& [best_x, best_y] = best_;

    for (std::size_t s = 0; s < steps; ++s) {
      acquisition_t acq{model_, best_y};
      optimizer_t opt{acq, config};

      const auto next_x = select_next_x(
          acq, opt, best_x, scan_samples, refine_top);

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
  // Cheaply score a batch of candidate points (no gradients),
  // then gradient-refine only the best `refine_top` of them,
  // keeping whichever refined optimum scores highest. This
  // gets most of the benefit of multi-start optimization
  // without paying full gradient-ascent cost per candidate.
  template <class AcqFn, class OptFn, class InputX>
  auto select_next_x(
      const AcqFn& acq, OptFn& opt, const InputX& best_x,
      std::size_t scan_samples,
      std::size_t refine_top) -> InputX {
    using scored_t = std::pair<number_t, InputX>;

    auto candidates = std::vector<scored_t>{};
    candidates.reserve(scan_samples + 1);
    candidates.emplace_back(acq(best_x), best_x);
    for (std::size_t i = 0; i < scan_samples; ++i) {
      auto x = (i % 2 == 0) ? domain_x_.random()
                             : domain_x_.generate(best_x);
      candidates.emplace_back(acq(x), std::move(x));
    }

    const auto top =
        std::min(refine_top, candidates.size());
    std::partial_sort(
        candidates.begin(),
        candidates.begin() + top,
        candidates.end(),
        [](const scored_t& a, const scored_t& b) {
          return a.first > b.first;
        });

    auto next_x = domain_x_.project(
        opt.maximize(candidates.front().second));
    auto next_score = acq(next_x);
    for (std::size_t i = 1; i < top; ++i) {
      const auto candidate_x = domain_x_.project(
          opt.maximize(candidates[i].second));
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
