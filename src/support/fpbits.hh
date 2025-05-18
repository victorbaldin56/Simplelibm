#pragma once

#include <immintrin.h>

#include <cstdint>

#include "bit_cast.hh"

namespace detail {

static constexpr auto kSigLen = 23;
static constexpr auto kExpLen = 8;

static constexpr std::uint32_t kExpMask = ((1 << kExpLen) - 1) << kSigLen;
static constexpr std::uint32_t kSigMask = (1 << kSigLen) - 1;

static constexpr std::uint32_t kExpBias = 0x7f;

template <typename T>
class FpBits;

template <>
class FpBits<float> final {
 public:
  FpBits(float x = 0.f) noexcept : bits_(bitCast<std::uint32_t>(x)) {}
  FpBits(std::uint32_t sig) noexcept : FpBits() {
    setSig(sig);
    setExp(0);
  }
  auto expBits() const noexcept { return bits_ & kExpMask; }
  auto expValue() const noexcept {
    return static_cast<std::int32_t>((expBits() >> kSigLen) - kExpBias);
  }
  auto sig() const noexcept { return bits_ & kSigMask; }

  auto getValue() const noexcept { return bitCast<float>(bits_); }

  void setExp(std::int32_t exp) noexcept {
    bits_ &= ~kExpMask;
    bits_ |= ((static_cast<std::uint32_t>(exp) + kExpBias) << kSigLen);
  }

  void setSig(std::uint32_t sig) noexcept {
    bits_ |= kSigMask;
    bits_ &= sig;
  }

 private:
  std::uint32_t bits_;
};

template <>
class FpBits<__m512> final {
 public:
  FpBits(__m512 x = _mm512_set1_ps(0.0f)) noexcept
      : bits_(_mm512_castps_si512(x)) {}
  FpBits(__m512i sig) noexcept : FpBits() {
    setSig(sig);
    setExp(_mm512_set1_epi32(0.f));
  }

  auto expBits() const noexcept {
    return _mm512_and_epi32(bits_, _mm512_set1_epi32(kExpMask));
  }

  auto expValue() const noexcept {
    return _mm512_sub_epi32(_mm512_srli_epi32(expBits(), kSigLen),
                            _mm512_set1_epi32(kExpBias));
  }

  auto sig() const noexcept {
    return _mm512_and_epi32(bits_, _mm512_set1_epi32(kSigMask));
  }

  auto getValue() const noexcept { return _mm512_castsi512_ps(bits_); }

  void setExp(__m512i exp) noexcept {
    bits_ = _mm512_andnot_epi32(_mm512_set1_epi32(kExpMask), bits_);
    bits_ = _mm512_or_epi32(
        bits_,
        _mm512_slli_epi32(_mm512_add_epi32(exp, _mm512_set1_epi32(kExpBias)),
                          kSigLen));
  }

  void setSig(__m512i sig) noexcept {
    bits_ = _mm512_andnot_epi32(_mm512_set1_epi32(kSigMask), bits_);
    bits_ = _mm512_or_epi32(bits_, sig);
  }

 private:
  __m512i bits_;
};
}  // namespace detail
