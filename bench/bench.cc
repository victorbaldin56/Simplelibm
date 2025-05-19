#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include "benchmark/benchmark.h"
#include "lalogf/logf.h"

constexpr std::uint32_t kMinBits = 0x00800000;  // 2^-126
constexpr std::uint32_t kMaxBits = 0x7f800000;  // +INF
constexpr std::uint32_t kNumPoints = kMaxBits - kMinBits;

void BM_LogfLatency(benchmark::State& state) {
  std::uint64_t total_cycles = 0;
  for (auto _ : state) {
    for (std::uint32_t bits = kMinBits; bits < kMaxBits; ++bits) {
      float x;
      std::memcpy(&x, &bits, 4);

      benchmark::DoNotOptimize(x);
      float result;

      auto start = __rdtsc();
      result = lalogf(x);
      auto end = __rdtsc();
      total_cycles += (end - start);
      benchmark::DoNotOptimize(result);
    }
  }
  double cpe =
      static_cast<double>(total_cycles) / (kNumPoints * state.iterations());
  state.counters["CPE"] =
      benchmark::Counter(cpe, benchmark::Counter::kDefaults);
}

void BM_LogfAvx512Throughput(benchmark::State& state) {
  std::uint64_t total_cycles = 0;
  const __m512i max_vec = _mm512_set1_epi32(kMaxBits);
  const __m512i vec_stride = _mm512_set1_epi32(16);
  for (auto _ : state) {
    __m512i bits = _mm512_setr_epi32(
        kMinBits, kMinBits + 1, kMinBits + 2, kMinBits + 3, kMinBits + 4,
        kMinBits + 5, kMinBits + 6, kMinBits + 7, kMinBits + 8, kMinBits + 9,
        kMinBits + 10, kMinBits + 11, kMinBits + 12, kMinBits + 13,
        kMinBits + 14, kMinBits + 15);
    while (_mm512_cmpge_epi32_mask(bits, max_vec) == 0) {
      __m512 x = _mm512_castps_si512(bits);
      benchmark::DoNotOptimize(x);
      __m512 result;

      auto start = __rdtsc();
      result = lalogf_avx512(x);
      auto end = __rdtsc();
      total_cycles += (end - start);
      benchmark::DoNotOptimize(result);
      bits = _mm512_add_epi32(bits, vec_stride);
    }
  }
  double cpe =
      static_cast<double>(total_cycles) / (kNumPoints * state.iterations());
  state.counters["CPE"] =
      benchmark::Counter(cpe, benchmark::Counter::kDefaults);
}

BENCHMARK(BM_LogfLatency)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_LogfAvx512Throughput)->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
