#pragma once

#include <algorithm>
#include <cmath>
#include <utility>

#include "dual/number.hpp"

namespace b2o::optimization {

template <class Number>
struct gradient_config {
  using number_t = Number;
  std::size_t lsteps{100};
  number_t lrate{1e-2};
  number_t leps{1e-6};
};
template <class Number>
gradient_config(std::size_t, Number, Number)
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
  auto minimize(Input input) const -> Input {
    auto dinput = dual::make_array(input);
    for (std::size_t s = 0; s < config_.lsteps; ++s) {
      const auto dresult = functor_(dinput);
      const auto& dvalue = dresult.dvalue();

      auto dmax = number_t{0};
      for (std::size_t i = 0; i < dinput.size(); ++i) {
        const auto dstep = config_.lrate * dvalue[i];
        dmax = std::max(dmax, std::abs(dstep));
        dinput[i] -= dstep;
      }
      if (dmax < config_.leps) {
        break;
      }
    }
    for (std::size_t i = 0; i < input.size(); ++i) {
      input[i] = dinput[i].value();
    }
    return input;
  }

 private:
  Functor functor_;
  config_t config_;
};

template <class Functor, class Number>
gradient(Functor, const gradient_config<Number>&)
    -> gradient<Functor, Number>;

}  // namespace b2o::optimization
