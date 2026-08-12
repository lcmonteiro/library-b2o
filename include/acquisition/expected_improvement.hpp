#pragma once

#include <algorithm>
#include <cmath>

#include "dual/operations/exp.hpp"
#include "dual/operations/log.hpp"
#include "dual/operations/minmax.hpp"
#include "dual/operations/sqrt.hpp"
#include "gaussian/distribution.hpp"

namespace b2o::acquisition {
/// =======================================================
/// Expected Improvement → Log(Expected Improvement)
/// Numerically Stable, Fully Smooth Derivation
/// (Minimization Case)
/// =======================================================
/// We assume:
///   f(x) ~ Normal(mu, sigma^2).
///   best = current minimum observed value .
///
/// Define:
///   delta = best - mu .
///   z     = delta / sigma .
/// -------------------------------------------------------
/// 1) Classical Expected Improvement (EI)
/// -------------------------------------------------------
/// EI(x) = E[max(best - f(x), 0)]
///
/// Closed form:
///   EI = delta * Φ(z) + sigma * φ(z) ,
///
/// where:
///   Φ(z) = standard normal CDF
///   φ(z) = standard normal PDF
/// -------------------------------------------------------
/// 2) Factor Out sigma  (critical step)
/// -------------------------------------------------------
/// Since:
///   delta = sigma * z ,
///
/// Rewrite:
///   EI = sigma * ( z Φ(z) + φ(z) ) .
/// -------------------------------------------------------
/// 3) Take Log
/// -------------------------------------------------------
///   log(EI) = log(sigma) + log( z Φ(z) + φ(z) ) ,
///
/// Focus on:
///   g(z) = z Φ(z) + φ(z) ,
///   g(z) > 0  for all real nummber.
/// -------------------------------------------------------
/// 4) Stabilize log(g(z))
/// -------------------------------------------------------
/// Rewrite:
///   g(z) = φ(z) * (1 + z Φ(z) / φ(z)) ,
///
/// Take log:
///   log g(z) = log φ(z) + log(1 + z Φ(z) / φ(z)) ,
///
/// Now express Φ(z)/φ(z) in log-space:
///   Φ(z) / φ(z) = exp( log Φ(z) - log φ(z) ) ,
///
/// Define:
///   lratio = log Φ(z) - log φ(z) ,
///   zratio = z * exp(lratio) ,
///
/// Then:
///   log g(z) = log φ(z) + log1p(r) ,
///
/// Final fully formula:
///   log(EI) = /
///   log(sigma) + /
///   log φ(z) + /
///   log1p[z * exp(log Φ(z) - log φ(z))] .
///
template <class Model, class Number>
class expected_improvement {
  static constexpr auto kJitter = Number{1e-12};
  static constexpr auto kMaxRatio = Number{1e20};
  static constexpr auto kMaxZ = Number{5};

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
    const auto best = compute_best(mu, sigma);
    const auto delta = best - mu;
    const auto z = delta / sigma;
    const auto lpdf = distribution_.log_pdf(z);
    const auto lcdf = distribution_.log_cdf(z);
    const auto lsigma = std::log(sigma);
    const auto lratio = lcdf - lpdf;
    const auto zratio = clamp_ratio(z * std::exp(lratio));
    return lsigma + lpdf + std::log1p(zratio);
  }

 protected:
  template <class T>
  auto compute_best(const T& mu, const T& sigma) const {
    return std::max(best_, mu - sigma * kMaxZ);
  }

  // Deep in the exploitation region (mu far below best_
  // relative to a small predictive sigma - e.g. right next to
  // a training point with a tight kernel lengthscale), z can
  // grow large enough that zratio overflows log1p. Saturate it
  // instead of letting that propagate as inf/NaN: past this
  // point log(EI) is already dominated by lsigma + lpdf anyway.
  template <class T>
  auto clamp_ratio(const T& zratio) const {
    return std::max(
        std::min(zratio, T{kMaxRatio}), T{-kMaxRatio});
  }

 private:
  const model_t& model_;
  const number_t best_;
  const distribution_t distribution_;
};

}  // namespace b2o::acquisition
