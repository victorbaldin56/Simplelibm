#include "lalogf/logf.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#include "gtest/gtest.h"
#include "multiprecision.hh"
#include "omp.h"

namespace {

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
  // iterate over whole testing range
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
  constexpr std::size_t kVectorSize = sizeof(__m512);
  constexpr std::size_t kElementsInVector = sizeof(__m512) / sizeof(float);
  double max_ulps = 0.0;

#pragma omp parallel for reduction(max : max_ulps)
  for (std::uint32_t bits = kMinBits; bits < kMaxBits;
       bits += kElementsInVector) {
    // prepare a vector of consecutive floats as an argument to vectorized logf
    std::array<std::uint32_t, kElementsInVector> bits_vector;
    std::iota(bits_vector.begin(), bits_vector.end(), bits);
    std::array<float, kElementsInVector> vals;
    std::memcpy(vals.data(), bits_vector.data(), kVectorSize);

    std::array<Highp, kElementsInVector> highp_vals;
    std::transform(vals.begin(), vals.end(), highp_vals.begin(),
                   [](float v) { return static_cast<Highp>(v); });

    std::array<Highp, kElementsInVector> highp_res;
    std::transform(vals.begin(), vals.end(), highp_res.begin(),
                   [](Highp v) { return log(v); });

    __m512 vec = _mm512_loadu_ps(vals.data());
    __m512 res = lalogf_avx512(vec);

    std::array<float, kElementsInVector> res_vec_arr;
    _mm512_storeu_ps(res_vec_arr.data(), res);
    std::array<double, kElementsInVector> ulps;
    std::transform(
        highp_res.begin(), highp_res.end(), res_vec_arr.begin(), ulps.begin(),
        [](Highp ref, float val) {
          float ulp = getUlp(val);
          return static_cast<double>(abs(ref - static_cast<Highp>(val)) /
                                     static_cast<Highp>(ulp));
        });
    double cur_max_ulps = *std::max_element(ulps.begin(), ulps.end());
    max_ulps = std::max(cur_max_ulps, max_ulps);
  }

  ASSERT_LE(max_ulps, kMaxUlps);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
