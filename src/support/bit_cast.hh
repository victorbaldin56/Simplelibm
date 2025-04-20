#pragma once

#include <cstring>
#include <type_traits>

namespace detail {

template <class To, class From>
std::enable_if_t<sizeof(To) == sizeof(From) &&
                     std::is_trivially_copyable_v<From> &&
                     std::is_trivially_copyable_v<To>,
                 To>
bitCast(const From& src) noexcept {
  static_assert(std::is_trivially_constructible_v<To>);
  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}
}
