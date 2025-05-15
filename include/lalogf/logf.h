#pragma once

#include <immintrin.h>

#ifdef __cplusplus
extern "C" {
#endif

float lalogf(float x);

__m512 lalogf_avx512(__m512 x);

#ifdef __cplusplus
}
#endif
