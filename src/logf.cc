#include <cerrno>
#include <limits>

#include "detail/fpbits.hh"
#include "math/math.h"

namespace {

constexpr float kLogfTable[] = {0.f};

}

float simpleLogf(float x) {
  if (x < 0) {
    errno = ERANGE;
    return std::numeric_limits<float>::quiet_NaN();
  }

  constexpr auto kLog2 = 0.69314718f;

  detail::FpBits bits(x);
  auto exp = bits.expValue();
  auto sig = bits.sig();

  bits.setExp(0);
  auto new_x = bits.getValue();

  auto index = (sig & (0xff << 15)) >> 15;
  auto r = kLogfTable[index];
}
