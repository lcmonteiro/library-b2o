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
  /// @brief Multi-start acquisition maximization.
  /// The acquisition is multi-modal with wide flat
  /// regions, so a single gradient ascent easily stalls
  /// in a poor local maximum. Instead:
  ///   1) screen many cheap candidates (plain numbers,
  ///      no autodiff): the incumbent, samples around
  ///      it (exploitation) and uniform samples
  ///      (exploration);
  ///   2) refine the best few by projected gradient
  ///      ascent and keep the overall maximum.
  template <class Input>
  auto propose(
      const acquisition_t& acq,  //
      const optimizer_t& opt,    //
      const Input& best_x) {
    using candidate_t = std::pair<number_t, Input>;
    const auto project = [this](const auto& x) {
      return domain_x_.project(x);
    };
    auto candidates = std::vector<candidate_t>{};
    candidates.reserve(kCandidates + 1);
    candidates.emplace_back(acq(best_x), best_x);
    for (std::size_t i = 0; i < kCandidates; ++i) {
      const auto x = i % 2 == 0 ? domain_x_.generate(best_x)
                                : domain_x_.random();
      candidates.emplace_back(acq(x), x);
    }
    const auto starts = std::min(kStarts, candidates.size());
    std::partial_sort(
        std::begin(candidates),
        std::begin(candidates) + starts,
        std::end(candidates),
        [](const auto& a, const auto& b) {
          return b.first < a.first;
        });
    auto [next_a, next_x] = candidates.front();
    for (std::size_t i = 0; i < starts; ++i) {
      const auto max_x =
          opt.maximize(candidates[i].second, project);
      const auto max_a = acq(max_x);
      if (next_a < max_a) {
        next_x = max_x;
        next_a = max_a;
      }
    }
    return next_x;
  }

 private:
  static constexpr std::size_t kCandidates = 1000;
  static constexpr std::size_t kStarts = 5;

  model_t model_;
  functor_t functor_;
  domain_x_t domain_x_;
  domain_y_t domain_y_;
  sample_t best_;
};

}  // namespace b2o::optimization::bayesian
