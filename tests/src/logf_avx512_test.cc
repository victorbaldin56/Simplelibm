#include <algorithm>
#include <cmath>
#include <cstring>

#include "gtest/gtest.h"
#include "lalogf/logf.h"

namespace {

constexpr unsigned kValuesPerSegment = 100;
}

TEST(lalogf_avx512, ulp) {
  for (int e = -126; e < 127; ++e) {
    auto cur = std::pow(2.f, static_cast<float>(e));
    auto stride = cur / kValuesPerSegment;
    for (unsigned i = 0; i < kValuesPerSegment; i += 16) {
      std::array<double, 16> vals;
      for (unsigned j = 0; j < 16; ++j) {
        vals[j] = cur + stride * (i + j);
      }
      std::array<double, 16> res;
      std::transform(vals.begin(), vals.end(), res.begin(),
                     [](auto v) { return std::log(v); });

      std::array<float, 16> float_vals;
      std::transform(vals.begin(), vals.end(), float_vals.begin(),
                     [](auto v) { return static_cast<float>(v); });
      auto vec = _mm512_loadu_ps(float_vals.data());
      auto res_vec = lalogf_avx512(vec);
      std::array<float, 16> res_vec_arr;
      _mm512_storeu_ps(res_vec_arr.data(), res_vec);

      std::array<double, 16> ulps;
      std::transform(res.begin(), res.end(), res_vec_arr.begin(), ulps.begin(),
                     [](auto ref, auto val) {
                       auto ulp =
                           std::nextafter(
                               val, std::numeric_limits<float>::infinity()) -
                           val;
                       return std::abs(ref - static_cast<double>(val)) / ulp;
                     });
      auto max_ulp = *std::max_element(ulps.begin(), ulps.end());
      ASSERT_LE(max_ulp, 1.5);
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
