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
// Final stage: user provides the objective function and
// builds the Bayesian optimizer.
// ============================================================
template <class Model, class Domain, class Functor>
class objective_builder {
  Model model_;
  Domain domain_;
  Functor functor_;

 public:
  // Takes ownership of model, domain, and functor
  objective_builder(Model m, Domain d, Functor f)
      : model_{std::move(m)},
        domain_{std::move(d)},
        functor_{std::move(f)} {
  }

  // Build the final Bayesian optimizer (consumes builder)
  auto build() && {
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
// User specifies the optimization domain (bounds,
// constraints).
// ============================================================
template <class Model, class Domain>
class domain_builder {
  Model model_;
  Domain domain_;

 public:
  // Takes ownership of model and domain
  domain_builder(Model m, Domain d)
      : model_{std::move(m)}, domain_{std::move(d)} {
  }

  // Custom objective function entry point
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
// User selects kernel and defines the domain.
// ============================================================
template <std::size_t Dimension, class Number, class Kernel>
class kernel_builder {
  Kernel kernel_;

 public:
  // Takes ownership of kernel
  explicit kernel_builder(Kernel&& k)
      : kernel_(std::move(k)) {
  }

  // Convenience domain bounds constructor
  template <class... Ts>
  auto domain_bounds(Ts&&... args) && {
    using bounds = domain::bounds<Number, Dimension>;
    return make_domain(bounds{std::forward<Ts>(args)...});
  }

  // Custom domain entry point (user-defined domains)
  template <class Domain>
  auto domain(Domain&& domain) && {
    return make_domain(std::forward<Domain>(domain));
  }

 protected:
  // Transition to domain stage
  template <class Domain>
  auto make_domain(Domain domain) {
    return domain_builder{
        gaussian::make_process<Dimension>(
            std::move(kernel_)),
        std::move(domain)};
  }
};

// ============================================================
// Stage 1: Optimizer Builder (Entry Point)
// User selects the kernel.
// ============================================================
template <std::size_t Dimension, class Number>
class optimizer_builder {
 public:
  // Convenience radial kernel constructor
  template <class T>
  auto kernel_radial(T&& sigma) {
    return make_kernel(
        kernel::radial{std::forward<T>(sigma)});
  }

  // Custom kernel entry point (user-defined kernels)
  template <class Kernel>
  auto kernel(Kernel&& kernel) {
    return make_kernel(std::forward<Kernel>(kernel));
  }

 protected:
  // Transition to kernel stage
  template <class Kernel>
  auto make_kernel(Kernel k) {
    return kernel_builder<Dimension, Number, Kernel>(
        std::move(k));
  }
};

// ============================================================
// Entry Function
// Factory for starting the fluent builder pipeline.
// ============================================================
template <std::size_t Dimension, class Number = double>
auto make_optimizer() {
  return optimizer_builder<Dimension, Number>{};
}

}  // namespace b2o
