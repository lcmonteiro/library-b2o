#pragma once

#include "dual/operations/sqrt.hpp"
#include "dual/operations/log.hpp"
#include "gaussian/distribution.hpp"

namespace b2o::acquisition {

template <class Model, class Number>
class expected_improvement {
  static constexpr auto kJitter = Number{1e-12};

 public:
  using distribution_t = gaussian::distribution<Number>;
  using number_t = Number;
  using model_t = Model;

  expected_improvement(const Model& model, Number best)
      : model_{model}, best_{best}, distribution_{} {
  }

  template <class Input>
  auto operator()(const Input& x) const {
    const auto [mu, var] = model_.predict(x);
    const auto sigma = std::sqrt(var + kJitter);
    const auto delta = best_ - mu;
    const auto z = delta / sigma;
    const auto cdf = distribution_.cdf(z);
    const auto pdf = distribution_.pdf(z);
    const auto ei = delta * cdf + sigma * pdf;
    return std::log(ei + kJitter);
  }

 private:
  const model_t& model_;
  const number_t best_;
  const distribution_t distribution_;
};

}  // namespace b2o::acquisition
