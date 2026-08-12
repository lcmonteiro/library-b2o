#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "dual/number.hpp"
#include "dual/operations.hpp"
#include "solver/cholesky.hpp"

namespace b2o::gaussian {

// ============================================================
// Isotropic radial-kernel lengthscale fitting
// ============================================================
// Picks the lengthscale `sigma` that maximizes the Gaussian
// Process log marginal likelihood on the given samples, via
// gradient ascent on theta = log(sigma) (keeps sigma positive
// throughout the search). `init_sigma` is only a starting
// guess for the search, not a fixed value.
//
// The marginal likelihood is extremely non-linear in theta: it
// is well-behaved near its peak but collapses sharply once the
// kernel matrix becomes near-singular (very large sigma), where
// the gradient magnitude explodes. A fixed-rate step there would
// overshoot into that unstable region, so each step's change to
// theta is capped at `max_step` regardless of gradient size -
// a simple trust-region safeguard - instead of reusing the
// uncapped optimization::gradient optimizer.
//
// Reuses math::cholesky with a dual-number kernel matrix so the
// marginal-likelihood gradient w.r.t. theta is obtained by
// forward-mode autodiff instead of a hand-derived formula.
template <class Number, class Samples>
auto fit_radial_lengthscale(
    const Samples& samples,  //
    Number noise,            //
    Number init_sigma,
    std::size_t steps = 60,
    Number max_step = Number{0.25},
    Number eps = Number{1e-4}) -> Number {
  const auto n = samples.size();
  if (n < 2)
    return init_sigma;

  const auto k_noise =
      std::max(noise * noise, Number{1e-12});

  // Bound the search to a sane range derived from the data's
  // own coordinate spread, instead of letting an unlucky small
  // sample pull the marginal likelihood toward a degenerate
  // lengthscale: too small collapses predictive variance to
  // near zero next to training points (blowing up downstream
  // acquisition-function ratios), too large makes the kernel
  // matrix near-singular.
  auto span = Number{0};
  const auto dim = samples.front().first.size();
  for (std::size_t d = 0; d < dim; ++d) {
    auto lo = samples.front().first[d];
    auto hi = lo;
    for (const auto& sample : samples) {
      lo = std::min(lo, sample.first[d]);
      hi = std::max(hi, sample.first[d]);
    }
    span += hi - lo;
  }
  span = std::max(span / Number(dim), Number{1e-3});
  const auto theta_lo = std::log(span * Number{0.05});
  const auto theta_hi = std::log(span * Number{10});

  const auto log_marginal_likelihood =
      [&](const dual::number<Number>& theta) {
        using dnumber_t = dual::number<Number>;

        const auto sigma = std::exp(theta);
        const auto denom = Number{2} * sigma * sigma;

        auto k = std::vector<std::vector<dnumber_t>>(n);
        for (std::size_t i = 0; i < n; ++i) {
          k[i].resize(n);
          const auto& xi = samples[i].first;
          for (std::size_t j = 0; j <= i; ++j) {
            const auto& xj = samples[j].first;
            auto sq = dnumber_t{Number{0}};
            for (std::size_t d = 0; d < xi.size(); ++d) {
              const auto delta = xi[d] - xj[d];
              sq = sq + delta * delta;
            }
            auto kij = std::exp(-(sq / denom));
            if (i == j)
              kij = kij + k_noise;
            k[i][j] = kij;
            k[j][i] = kij;
          }
        }

        const auto solver = math::cholesky<dnumber_t>{};
        auto l = std::vector<std::vector<dnumber_t>>{};
        solver.build(k, l);

        auto y = std::vector<Number>(n);
        for (std::size_t i = 0; i < n; ++i)
          y[i] = samples[i].second;

        auto z = std::vector<dnumber_t>{};
        solver.forward(l, y, z);

        auto data_fit = dnumber_t{Number{0}};
        auto log_det = dnumber_t{Number{0}};
        for (std::size_t i = 0; i < n; ++i) {
          data_fit =
              data_fit - Number{0.5} * z[i] * z[i];
          log_det = log_det - std::log(l[i][i]);
        }
        return data_fit + log_det;
      };

  auto theta = std::clamp(
      std::log(init_sigma), theta_lo, theta_hi);
  for (std::size_t s = 0; s < steps; ++s) {
    const auto mll =
        log_marginal_likelihood(dual::make_number(theta));
    const auto grad = mll.dvalue(0);
    const auto delta =
        std::clamp(grad, -max_step, max_step);
    if (std::abs(delta) < eps)
      break;
    theta = std::clamp(
        theta + delta, theta_lo, theta_hi);
  }
  return std::exp(theta);
}

}  // namespace b2o::gaussian
