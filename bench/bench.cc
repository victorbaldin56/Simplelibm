#include <immintrin.h>

#include <cmath>
#include <cstdint>

#include "benchmark/benchmark.h"

static void BM_LogfLatency(benchmark::State& state) {
  float x = 1.2345f;

  std::uint64_t total_cycles = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    float result;

    auto start = __rdtsc();
    result = logf(x);
    auto end = __rdtsc();
    total_cycles += (end - start);

    benchmark::DoNotOptimize(result);
  }
  double cpe = static_cast<double>(total_cycles) / state.iterations();
  state.counters["CPE"] =
      benchmark::Counter(cpe, benchmark::Counter::kDefaults);
}

// Register the benchmark
BENCHMARK(BM_LogfLatency)
    ->Unit(benchmark::kNanosecond)
    ->Repetitions(8)
    ->ReportAggregatesOnly(true);

BENCHMARK_MAIN();
