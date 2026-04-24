#pragma once

#include <algorithm>
#include <cmath>

#include "dual/operations/exp.hpp"
#include "dual/operations/log.hpp"
#include "dual/operations/minmax.hpp"
#include "dual/operations/sqrt.hpp"
#include "gaussian/distribution.hpp"
#include "helpers/print.hpp"

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
    // print_number("best_", best_);
    // print_number("best", best);
    // print_number("mu", mu);
    // print_number("sigma", sigma);
    const auto delta = best - mu;
    const auto z = delta / sigma;
    // print_number("z", z);
    return std::log(
        delta * distribution_.cdf(z) +
        sigma * distribution_.pdf(z));

    /*
    const auto [mu, var] = model_.predict(x);
    const auto sigma = std::sqrt(var + kJitter);
    const auto best =
        compute_best(mu.value(), sigma.value());
    print_number("best_", best_);
    print_number("best", best);
    print_number("mu", mu);
    print_number("sigma", sigma);
    const auto delta = best - mu;
    const auto z = delta / sigma;

    print_number("z", z);
    const auto lpdf = distribution_.log_pdf(z);
    print_number("lpdf", lpdf);

    const auto lcdf = distribution_.log_cdf(z);
    print_number("lcdf", lcdf);
    const auto lsigma = std::log(sigma);
    print_number("lsigma", lsigma);
    const auto lratio = lcdf - lpdf;
    print_number("lratio", lratio);
    print_number("eratio", std::exp(lratio));
    const auto zratio = z * std::exp(lratio);
    assert(zratio < decltype(zratio){kMaxRatio});
    print_number("zratio", zratio);
    const auto lei = lsigma + lpdf + std::log1p(zratio);
    return lei;
    */
    /*
        const auto sigma = std::sqrt(var + kJitter);
        print_number("sigma", sigma);
        const auto delta = best_ - mu;
        print_number("delta", delta);
        const auto z = delta / sigma;
        print_number("z", z);
        const auto pdf_log = distribution_.log_pdf(z);
        print_number("pdf_log", pdf_log);
        const auto pdf_weight = std::log(sigma);
        const auto pdf_term = pdf_weight + pdf_log;
        print_number("pdf_term", pdf_term);
        if (delta < decltype(delta){kJitter}) {
          return pdf_term; e
        }
        const auto cdf_log = distribution_.log_cdf(z);
        print_number("cdf_log", cdf_log);
        const auto cdf_weight = std::log(delta);
        const auto cdf_term = cdf_weight + cdf_log;
        print_number("cdf_term", cdf_term);
        const auto max_term = std::max(cdf_term,
       pdf_term); print_number("max_term", max_term);
        const auto cdf_exp = std::exp(cdf_term -
       max_term); print_number("cdf_exp", cdf_exp); const
       auto pdf_exp = std::exp(pdf_term - max_term);
        print_number("pdf_exp", pdf_exp);
        const auto lei = max_term + std::log(cdf_exp +
       pdf_exp); print_number("lei", lei); return lei;
        */
  }

 protected:
  auto compute_best(Number mu, Number sigma) const {
    return std::max(best_, mu - sigma * kMaxZ);
  }

 private:
  const model_t& model_;
  const number_t best_;
  const distribution_t distribution_;
};

}  // namespace b2o::acquisition
