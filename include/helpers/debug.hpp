#pragma once
#include <array>
#include <fstream>
#include <sstream>
#include <string>

class debug_3d final {
  static constexpr auto kMin = -5.0;
  static constexpr auto kMax = 15.0;
  static constexpr auto kStp = 1.0;

 public:
  debug_3d(std::string path) : os_{path} {
    os_ << "[" << std::endl;
  }

  ~debug_3d() {
    os_.seekp(-2, std::ios_base::end);
    os_ << std::endl << "]" << std::endl;
  }

  template <class S, class M>
  auto print(const S& s, const M& m) {
    os_ << "{ " << std::endl;
    print_sample("s", s);
    print_model("m", m);
    os_.seekp(-2, std::ios_base::end);
    os_ << std::endl << "}," << std::endl;
  }

  template <class S, class M, class A>
  auto print(const S& s, const M& m, const A& a) {
    os_ << "{ " << std::endl;
    print_sample("s", s);
    print_model("m", m);
    print_function("a", a);
    os_.seekp(-2, std::ios_base::end);
    os_ << std::endl << "}," << std::endl;
  }

 protected:
  template <class Prefix, class Model>
  auto print_model(const Prefix& p, const Model& m) {
    std::stringstream sx;
    std::stringstream sy;
    std::stringstream sz;
    std::stringstream ss;
    sx << "\""<< p << "_x" << "\":" << "[ ";
    sy << "\""<< p << "_y" << "\":" << "[ ";
    sz << "\""<< p << "_z" << "\":" << "[ ";
    ss << "\""<< p << "_s" << "\":" << "[ ";
    for (auto x = kMin; x < kMax; x += kStp) {
      for (auto y = kMin; y < kMax; y += kStp) {
        auto [z, s] = m.predict(std::array{x, y});
        sx << x << ", ";
        sy << y << ", ";
        sz << z << ", ";
        ss << s << ", ";
      }
    }
    sx.seekp(-2, std::ios_base::end);
    sy.seekp(-2, std::ios_base::end);
    sz.seekp(-2, std::ios_base::end);
    ss.seekp(-2, std::ios_base::end);
    sx << " ]," << std::endl;
    sy << " ]," << std::endl;
    sz << " ]," << std::endl;
    ss << " ]," << std::endl;
    os_ << sx.str() << sy.str() << sz.str() << ss.str();
  }

  template <class Prefix, class Function>
  auto print_function(const Prefix& p, const Function& f) {
    std::stringstream sx;
    std::stringstream sy;
    std::stringstream sz;
    sx << "\""<< p << "_x" << "\":" << "[ ";
    sy << "\""<< p << "_y" << "\":" << "[ ";
    sz << "\""<< p << "_z" << "\":" << "[ ";
    for (auto x = kMin; x < kMax; x += kStp) {
      for (auto y = kMin; y < kMax; y += kStp) {
        auto z = f(std::array{x, y});
        sx << x << ", ";
        sy << y << ", ";
        sz << z << ", ";
      }
    }
    sx.seekp(-2, std::ios_base::end);
    sy.seekp(-2, std::ios_base::end);
    sz.seekp(-2, std::ios_base::end);
    sx << " ]," << std::endl;
    sy << " ]," << std::endl;
    sz << " ]," << std::endl;
    os_ << sx.str() << sy.str() << sz.str();
  }

  template <class Prefix, class Sample>
  auto print_sample(const Prefix& p, const Sample& s) {
    auto& [u, z] = s;
    auto& [x, y] = u;
    os_ << "\""<< p << "_x" << "\":" << x << "," << std::endl;
    os_ << "\""<< p << "_y" << "\":" << y << "," << std::endl;
    os_ << "\""<< p << "_z" << "\":" << z << "," << std::endl;
  }

 private:
  std::ofstream os_;
};
