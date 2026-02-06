#pragma once
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "acquisition/expected_improvement.hpp"
#include "domain/bounds.hpp"
#include "gaussian/process.hpp"
#include "kernel/radial.hpp"
#include "optimization/bayesian.hpp"
#include "optimization/gradient.hpp"

namespace b2o {

template <
    std::size_t D, class N,  //
    class Model, class Domain, class Functor>
class optimizer_builder {
  Model model_;
  Domain domain_;
  Functor functor_;

 public:
  optimizer_builder(Model m, Domain d, Functor f)
      : model_(std::move(m)),
        domain_(std::move(d)),
        functor_(std::move(f)) {
  }

  auto build() const&& {
    return optimization::bayesian<
        acquisition::expected_improvement,
        optimization::gradient,
        Model Domain,
        Functor>{
        std::move(model_),
        std::move(domain_),
        std::move(functor_)};
  }
};

template <std::size_t D, class N, class Model, class Domain>
class domain_builder {
  Model model_;
  Domain domain_;

 public:
  domain_builder(Model m, Domain d)
      : model_(std::move(m)), domain_(d) {
  }

  template <class Functor>
  auto objective(Functor&& fn) {
    return builder<D, K, Obj>{
        model_, domain_, std::forward<Functor>(fn)};
  }
};

template <std::size_t D, class N, class Kernel>
class kernel_builder {
  Kernel kernel_;

 public:
  explicit kernel_builder(Kernel&& k)
      : kernel_(std::move(k)) {
  }

  template <class... Ts>
  auto domain_bounds(Ts&&... args) const&& {
    return domain_builder(
        gaussian::make_process(std::move(kernel_)),
        domain::bounds<N, D>{std::forward<Ts>(args)...});
  }

 protected:
  template <class Model, class Domain>
  auto make_domain(Model m, Domain d) {
    return domain_builder<D, N, Model, Domain>{m, d};
  }
};

template <std::size_t D, class N>
class optimizer_builder {
 public:
  auto kernel_radial(N sigma) {
    return make_kernel(kernel::radial{sigma});
  }

 protected:
  template <class K>
  auto make_kernel(K k) {
    return kernel_builder<D, N, K>(std::move(k));
  }
};

template <std::size_t Dimension, class Number = double>
auto make_optimizer() {
  return optimizer_builder<Dimension, Number>{};
}

}  // namespace b2o
