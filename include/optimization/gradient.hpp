#pragma once

#include <cstddef>
#include <utility>

#include "dual/number.hpp"

namespace b2o::optimization {

template <class Number>
struct gradient_config {
  using number_t = Number;

  number_t lrate;
  std::size_t lsteps;
};
template <class Number>
gradient_config(Number, std::size_t)
    -> gradient_config<Number>;

template <class Functor, class Number>
class gradient {
 public:
  using number_t = Number;
  using config_t = gradient_config<Number>;

  template <class Fn>
  explicit gradient(Fn&& functor, const config_t& config)
      : functor_{std::forward<Fn>(functor)},
        config_{config} {
  }

  template <class Input>
  auto minimize(Input x) const -> Input {
    for (std::size_t s = 0; s < config_.lsteps; ++s) {
      const auto xd = dual::make_array(x);
      const auto yd = functor_(xd);
      const auto dvalue = yd.dvalue();
      for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] -= config_.lrate * dvalue[i];
      }
    }
    return x;
  }

 private:
  Functor functor_;
  config_t config_;
};

template <class Functor, class Number>
gradient(Functor, const gradient_config<Number>&)
    -> gradient<Functor, Number>;

}  // namespace b2o::optimization
