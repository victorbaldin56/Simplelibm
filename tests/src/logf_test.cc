#include <algorithm>
#include <cmath>

#include "gtest/gtest.h"
#include "lalogf/logf.h"

namespace {

constexpr unsigned kValuesPerSegment = 1000000;
static_assert((kValuesPerSegment & 0xf) == 0);

constexpr double kMaxUlps = 3.5;

float getUlp(float x) {
  return std::nextafter(x, std::numeric_limits<float>::infinity()) - x;
}
}  // namespace

TEST(lalogf, ulp) {
  for (int e = -126; e < 127; ++e) {
    float cur = std::pow(2.0, e);
    float stride = cur / kValuesPerSegment;
    for (unsigned i = 0; i < kValuesPerSegment; i += 16) {
      float val = cur + stride * i;
      double ref = std::log(static_cast<double>(val));
      float res = lalogf(val);
      float ulp = getUlp(res);
      double ulp_error =
          std::abs(ref - static_cast<double>(res)) / static_cast<double>(ulp);
      ASSERT_LE(ulp_error, kMaxUlps);
    }
  }
}

TEST(lalogf_avx512, ulp) {
  for (int e = -126; e < 127; ++e) {
    float cur = std::pow(2.f, static_cast<float>(e));
    float stride = cur / kValuesPerSegment;
    for (unsigned i = 0; i < kValuesPerSegment; i += 16) {
      std::array<float, 16> vals;
      for (unsigned j = 0; j < 16; ++j) {
        vals[j] = cur + stride * (i + j);
      }

      std::array<double, 16> double_vals;
      std::transform(vals.begin(), vals.end(), double_vals.begin(),
                     [](float v) { return static_cast<double>(v); });
      std::array<double, 16> res;
      std::transform(vals.begin(), vals.end(), res.begin(),
                     [](auto v) { return std::log(v); });

      __m512 vec = _mm512_loadu_ps(vals.data());
      __m512 res_vec = lalogf_avx512(vec);
      std::array<float, 16> res_vec_arr;
      _mm512_storeu_ps(res_vec_arr.data(), res_vec);
      std::array<double, 16> ulps;
      std::transform(res.begin(), res.end(), res_vec_arr.begin(), ulps.begin(),
                     [](double ref, float val) {
                       float ulp = getUlp(val);
                       return std::abs(ref - static_cast<double>(val)) /
                              static_cast<double>(ulp);
                     });
      double max_ulp = *std::max_element(ulps.begin(), ulps.end());
      ASSERT_LE(max_ulp, kMaxUlps);
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
