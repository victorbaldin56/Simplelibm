#pragma once

#include <cstdint>

#include "bit_cast.hh"

namespace detail {

class FpBits {
 public:
  FpBits(float x = 0.f) noexcept : bits_(bitCast<std::uint32_t>(x)) {}
  FpBits(std::uint32_t sig) noexcept : FpBits() { setSig(sig); }
  auto expBits() const noexcept { return bits_ & kExpMask; }
  auto expValue() const noexcept {
    return static_cast<std::int32_t>(expBits() >> kSigLen) - kExpBias;
  }
  auto sig() const noexcept { return bits_ & kSigMask; }

  auto getValue() const noexcept { return bitCast<float>(bits_); }

  void setExp(std::uint32_t exp) noexcept {
    bits_ |= kExpMask;
    bits_ &= (exp << kSigLen);
  }

  void setSig(std::uint32_t sig) noexcept {
    bits_ |= kSigMask;
    bits_ &= sig;
  }

 public:
  static constexpr auto kSigLen = 23;
  static constexpr auto kExpLen = 8;

  static constexpr std::uint32_t kExpMask = ((1 << kExpLen) - 1) << kSigLen;
  static constexpr std::uint32_t kSigMask = (1 << kSigLen) - 1;

  static constexpr std::int32_t kExpBias = 0x7f;

 private:
  std::uint32_t bits_;
};
}  // namespace detail
