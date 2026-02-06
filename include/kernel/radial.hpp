#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <tuple>

namespace b2o::kernel {

template <class Number>
class radial {
  static constexpr auto two = Number{2.0};
  static constexpr auto zero = Number{0.0};

 public:
  using number_t = Number;

  explicit radial(const Number& sigma)
      : denominator_{two * sigma * sigma} {
    assert(sigma > zero);
  }

  template <class SampleX, class SampleY>
  auto operator()(
      const SampleX& x,  //
      const SampleY& y) const {
    constexpr auto nx = std::tuple_size_v<SampleX>;
    constexpr auto ny = std::tuple_size_v<SampleY>;
    static_assert(nx == ny);
    return compute(x, y, std::make_index_sequence<nx>{});
  }

 protected:
  template <class SampleX, class SampleY, size_t... I>
  auto compute(
      const SampleX& x,  //
      const SampleY& y,  //
      std::index_sequence<I...>) const {
    return std::exp(-(([&] {
      const auto delta = std::get<I>(x) - std::get<I>(y);
      return (delta * delta) / denominator_;
    }() + ...)));
  }

 private:
  const number_t denominator_{};
};

template <class Number, std::size_t N>
class radial<std::array<Number, N> > {
  static constexpr auto two = Number{2.0};
  static constexpr auto zero = Number{0.0};
  static constexpr auto init = [](auto... sigma) {
    ((assert(sigma > zero)), ...);
    return std::array{(two * sigma * sigma)...};
  };

 public:
  using number_t = Number;

  explicit radial(const std::array<Number, N>& sigma)
      : denominator_{std::apply(init, sigma)} {
  }

  template <class SampleX, class SampleY>
  auto operator()(
      const SampleX& x,  //
      const SampleY& y) const {
    constexpr auto nx = std::tuple_size_v<SampleX>;
    constexpr auto ny = std::tuple_size_v<SampleY>;
    static_assert(nx == N, "dimension x mismatch");
    static_assert(ny == N, "dimension y mismatch");
    return compute(x, y, std::make_index_sequence<nx>{});
  }

 protected:
  template <class SampleX, class SampleY, size_t... I>
  auto compute(
      const SampleX& x,  //
      const SampleY& y,  //
      std::index_sequence<I...>) const {
    return std::exp(-(([&] {
      const auto delta = std::get<I>(x) - std::get<I>(y);
      const auto denominator = std::get<I>(denominator_);
      return (delta * delta) / denominator;
    }() + ...)));
  }

 private:
  const std::array<number_t, N> denominator_{};
};

}  // namespace b2o::kernel
