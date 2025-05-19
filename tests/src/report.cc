#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "fmt/core.h"
#include "fmt/format.h"
#include "lalogf/logf.h"
#include "multiprecision.hh"

namespace {

auto constructOfstream(const std::filesystem::path& path) {
  std::ofstream s;
  s.exceptions();
  s.open(path);
  return s;
}
}

int main() try {
  const auto reports_dir =
      std::filesystem::absolute(__FILE__).parent_path().parent_path() /
      "reports";
  const auto logf_report_path = reports_dir / "lalogf.csv";
  const auto logf_avx512_report_path = reports_dir / "lalogf_avx512.csv";

  auto logf_report = constructOfstream(logf_report_path);
  auto logf_avx512_report = constructOfstream(logf_avx512_report_path);

  const std::string csv_header = "x,y,abs_err,rel_err,first_error_bit\n";
  logf_report << csv_header;
  logf_avx512_report << csv_header;

  constexpr std::size_t kNumSamples = 100000;
  constexpr float kMin = 0.0008f;
  constexpr float kMax = 8.f;
  constexpr float kStride = (kMax - kMin) / kNumSamples;

  for (float x = kMin; x < kMax; x += kStride) {
    float res = lalogf(x);
    Highp res_highp = log(static_cast<Highp>(x));
    double abs_error =
        static_cast<double>(abs(res_highp - static_cast<Highp>(res)));
    double rel_error = std::abs(abs_error / static_cast<double>(res_highp));

    float res_highp_f = static_cast<float>(res_highp);
    std::uint32_t bits, bits_highp;
    std::memcpy(&bits, &res, 4);
    std::memcpy(&bits_highp, &res_highp_f, 4);
    std::uint32_t diff = bits ^ bits_highp;

    std::uint32_t first_bit = (diff ? __builtin_clz(diff) - 8 : -1);
    logf_report << fmt::format("{:.8e},{:.8e},{:.8e},{:.8e},{}\n", x, res,
                               abs_error, rel_error, first_bit);
  }

  constexpr std::size_t kVectorSize = sizeof(__m512);
  constexpr std::size_t kElementsInVector = sizeof(__m512) / sizeof(float);
  for (float x = kMin; x < kMax; x += kElementsInVector * kStride) {
    __m512 xs{
        x + kStride * 0,  x + kStride * 1,  x + kStride * 2,  x + kStride * 3,
        x + kStride * 4,  x + kStride * 5,  x + kStride * 6,  x + kStride * 7,
        x + kStride * 8,  x + kStride * 9,  x + kStride * 10, x + kStride * 11,
        x + kStride * 12, x + kStride * 13, x + kStride * 14, x + kStride * 15};
    __m512 res = lalogf_avx512(xs);

    std::array<float, kElementsInVector> res_elems;
    _mm512_storeu_ps(res_elems.data(), res);
    std::array<float, kElementsInVector> xs_elems;
    _mm512_storeu_ps(xs_elems.data(), xs);

    std::array<Highp, kElementsInVector> xs_highp;
    std::transform(xs_elems.begin(), xs_elems.end(), xs_highp.begin(),
                   [](float v) { return static_cast<Highp>(v); });
    std::array<Highp, kElementsInVector> res_highp;
    std::transform(xs_highp.begin(), xs_highp.end(), res_highp.begin(),
                   [](Highp v) { return log(v); });

    std::array<double, kElementsInVector> abs_error;
    std::transform(
        res_highp.begin(), res_highp.end(), res_elems.begin(),
        abs_error.begin(), [](Highp ref, float res) {
          return static_cast<double>(abs(ref - static_cast<Highp>(res)));
        });

    std::array<double, kElementsInVector> rel_error;
    std::transform(res_highp.begin(), res_highp.end(), abs_error.begin(),
                   rel_error.begin(), [](Highp ref, double abs_error) {
                     return std::abs(abs_error / static_cast<double>(ref));
                   });

    std::array<float, kElementsInVector> res_highp_f;
    std::transform(res_highp.begin(), res_highp.end(), res_highp_f.begin(),
                   [](Highp ref) { return static_cast<float>(ref); });
    __m512i bits_highp = _mm512_loadu_epi32(res_highp_f.data());
    __m512i bits = _mm512_castps_si512(res);
    __m512i diff = _mm512_xor_epi32(bits, bits_highp);

    __mmask16 zero_m = _mm512_cmpeq_epi32_mask(diff, _mm512_set1_epi32(0));
    __m512i pos =
        _mm512_sub_epi32(_mm512_lzcnt_epi32(diff), _mm512_set1_epi32(8));
    __m512i first_bit =
        _mm512_mask_mov_epi32(pos, zero_m, _mm512_set1_epi32(-1));
    std::array<std::uint32_t, kElementsInVector> first_bits;
    _mm512_storeu_epi32(first_bits.data(), first_bit);

    for (std::size_t j = 0; j < kElementsInVector; ++j) {
      logf_avx512_report << fmt::format("{:.8e},{:.8e},{:.8e},{:.8e},{}\n",
                                        xs_elems[j], res_elems[j], abs_error[j],
                                        rel_error[j], first_bits[j]);
    }
  }

  return 0;
} catch (std::runtime_error& e) {
  std::cerr << e.what() << std::endl;
  return EXIT_FAILURE;
}
