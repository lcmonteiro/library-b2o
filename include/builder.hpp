#pragma once
#include <cstddef>
#include <utility>

#include "acquisition/expected_improvement.hpp"
#include "domain/bounds.hpp"
#include "gaussian/process.hpp"
#include "kernel/radial.hpp"
#include "optimization/bayesian.hpp"
#include "optimization/gradient.hpp"

namespace b2o {

// ============================================================
// Stage 4: Objective Builder
// ============================================================
template <class Model, class Domain, class Functor>
class objective_builder {
  Model model_;
  Domain domain_;
  Functor functor_;

 public:
  objective_builder(Model m, Domain d, Functor f)
      : model_{std::move(m)},
        domain_{std::move(d)},
        functor_{std::move(f)} {
  }

  // Build the final Bayesian optimizer
  auto build() const&& {
    return optimization::bayesian<
        acquisition::expected_improvement,
        optimization::gradient,
        Model,
        Domain,
        Functor>{
        std::move(model_),
        std::move(domain_),
        std::move(functor_)};
  }
};

// ============================================================
// Stage 3: Domain Builder
// ============================================================
template <class Model, class Domain>
class domain_builder {
  Model model_;
  Domain domain_;

 public:
  domain_builder(Model m, Domain d)
      : model_{std::move(m)}, domain_{std::move(d)} {
  }

  // Attach the objective function
  template <class Functor>
  auto objective(Functor fn) && {
    return objective_builder<Model, Domain, Functor>{
        std::move(model_),
        std::move(domain_),
        std::move(fn)};
  }
};

// ============================================================
// Stage 2: Kernel Builder
// ============================================================
template <std::size_t Dimension, class Number, class Kernel>
class kernel_builder {
  Kernel kernel_;

 public:
  explicit kernel_builder(Kernel&& k)
      : kernel_(std::move(k)) {
  }

  // Define domain bounds
  template <class... Ts>
  auto domain_bounds(Ts&&... args) && {
    return domain_builder{
        gaussian::make_process<Dimension>(
            std::move(kernel_)),
        domain::bounds<Number, Dimension>{
            std::forward<Ts>(args)...}};
  }
};

// ============================================================
// Stage 1: Optimizer Builder (entry point)
// ============================================================
template <std::size_t Dimension, class Number>
class optimizer_builder {
 public:
  // Predefined radial kernel
  template <class T>
  auto kernel_radial(T&& sigma) {
    return make_kernel(
        kernel::radial{std::forward<T>(sigma)});
  }

 protected:
  // Generic kernel hook
  template <class Kernel>
  auto make_kernel(Kernel k) {
    return kernel_builder<Dimension, Number, Kernel>(
        std::move(k));
  }
};

// ============================================================
// Entry Function
// ============================================================
template <std::size_t Dimension, class Number = double>
auto make_optimizer() {
  return optimizer_builder<Dimension, Number>{};
}

}  // namespace b2o
