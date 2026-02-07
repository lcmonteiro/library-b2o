#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <iterator>
#include <random>

namespace b2o::domain {

template <class Number, std::size_t N>
class bounds {
 public:
  using number_t = Number;
  using input_t = std::array<number_t, N>;
  using range_t = std::pair<number_t, number_t>;
  using config_t = std::array<range_t, N>;
  using generator_t = std::mt19937;

  bounds(const config_t& config, const input_t& start)
      : config_{config}, start_{project(start)}, rng_{} {
    rng_.seed(std::random_device{}());
  }

  template <class T>
  auto seed(T&& value) -> void {
    rng_.seed(std::forward<T>(value));
  }

  auto start() const -> const input_t& {
    return start_;
  }

  auto random() -> input_t {
    return build({}, [this](auto, auto lo, auto hi) {
      std::uniform_real_distribution<Number> dist(lo, hi);
      return dist(rng_);
    });
  }

  auto generate(const input_t& center) -> input_t {
    constexpr auto zero = Number{0};
    constexpr auto one = Number{1};
    return build(center, [this](auto x, auto lo, auto hi) {
      const auto points = std::array{lo, x, hi};
      const auto weights = std::array{zero, one, zero};
      std::piecewise_linear_distribution<Number> dist{
          std::cbegin(points),
          std::cend(points),
          std::cbegin(weights)};
      return dist(rng_);
    });
  }

  auto project(const input_t& value) const -> input_t {
    return build(value, [](auto x, auto lo, auto hi) {
      return std::clamp(x, lo, hi);
    });
  }

 protected:
  template <class Fn>
  auto build(const input_t& value, Fn fn) const -> input_t {
    input_t result{};
    std::transform(
        std::cbegin(value),
        std::cend(value),
        std::cbegin(config_),
        std::begin(result),
        [&fn](const auto& x, const auto& limits) {
          const auto& [lo, hi] = limits;
          return std::invoke(fn, x, lo, hi);
        });
    return result;
  }

 private:
  config_t config_;
  input_t start_;
  generator_t rng_;
};

}  // namespace b2o::domain
