#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace b2o::domain {

template <class Number>
class normalize_builder {
 public:
  static constexpr auto kMinSigma = Number{1e-6};

  class normalize final {
   public:
    auto project(Number value) const -> Number {
      return (value - mean_) / sigma_;
    }

    auto inverse(Number norm) const -> Number {
      return norm * sigma_ + mean_;
    }

   protected:
    friend class normalize_builder<Number>;

    normalize(Number mean, Number sigma)
        : mean_{mean}, sigma_{sigma} {
      if (sigma_ <= Number{1e-9})
        throw std::invalid_argument(
            "Standard deviation must be positive");
    }

   private:
    Number mean_;
    Number sigma_;
  };

  auto emplace(Number value) -> void {
    cache_.emplace_back(value);
  }

  auto build() const -> normalize {
    if (cache_.empty())
      throw std::runtime_error(
          "Cannot build: no data in cache");

    const auto mean = compute_mean();
    const auto sigma = compute_sigma(mean);
    return normalize{mean, sigma};
  }

 protected:
  auto compute_mean() const -> Number {
    assert(!cache_.empty());
    const auto sum = std::accumulate(
        std::cbegin(cache_), std::cend(cache_), Number(0));
    return sum / static_cast<Number>(cache_.size());
  }

  auto compute_sigma(Number mean) const -> Number {
    assert(!cache_.empty());
    const auto sum_sq = std::transform_reduce(
        std::cbegin(cache_),
        std::cend(cache_),
        Number{0},
        std::plus<>(),
        [mean](Number v) {
          const auto d = v - mean;
          return d * d;
        });
    const auto sigma = std::sqrt(
        sum_sq / static_cast<Number>(cache_.size()));
    return std::max(sigma, kMinSigma);
  }

 private:
  std::vector<Number> cache_;
};
}  // namespace b2o::domain
