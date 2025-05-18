#include "lalogf/logf.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

#include "boost/multiprecision/cpp_bin_float.hpp"
#include "gtest/gtest.h"
#include "omp.h"

namespace {

using Highp = boost::multiprecision::cpp_bin_float_double;

constexpr unsigned kValuesPerSegment = 1000000;
static_assert((kValuesPerSegment & 0xf) == 0);

constexpr double kMaxUlps = 1.0;

constexpr std::uint32_t kMinBits = 0x00800000;  // 2^-126
constexpr std::uint32_t kMaxBits = 0x7f800000;  // +INF

float getNext(float x) {
  return std::nextafter(x, std::numeric_limits<float>::infinity());
}

float getUlp(float x) { return getNext(x) - x; }
}  // namespace

TEST(lalogf, ulp) {
  double max_ulps = 0.0;

#pragma omp parallel for reduction(max : max_ulps)
  for (std::uint32_t bits = kMinBits; bits < kMaxBits; ++bits) {
    float val;
    std::memcpy(&val, &bits, 4);
    Highp ref = log(static_cast<Highp>(val));
    float res = lalogf(val);
    float ulp = getUlp(res);
    double ulp_error = static_cast<double>(abs(ref - static_cast<Highp>(res)) /
                                           static_cast<Highp>(ulp));
    max_ulps = std::max(max_ulps, ulp_error);
  }

  ASSERT_LE(max_ulps, kMaxUlps);
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

      std::array<Highp, 16> highp_vals;
      std::transform(vals.begin(), vals.end(), highp_vals.begin(),
                     [](float v) { return static_cast<Highp>(v); });
      std::array<Highp, 16> res;
      std::transform(vals.begin(), vals.end(), res.begin(),
                     [](Highp v) { return log(v); });

      __m512 vec = _mm512_loadu_ps(vals.data());
      __m512 res_vec = lalogf_avx512(vec);
      std::array<float, 16> res_vec_arr;
      _mm512_storeu_ps(res_vec_arr.data(), res_vec);
      std::array<double, 16> ulps;
      std::transform(res.begin(), res.end(), res_vec_arr.begin(), ulps.begin(),
                     [](Highp ref, float val) {
                       float ulp = getUlp(val);
                       return static_cast<double>(
                           abs(ref - static_cast<Highp>(val)) /
                           static_cast<Highp>(ulp));
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
