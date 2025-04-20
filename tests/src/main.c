#include <stdio.h>
#include <stdlib.h>

#include "simple/math.h"

int main() {
  float x;
  if (scanf("%f", &x) != 1) {
    fprintf(stderr, "Invalid input format\n");
    return EXIT_FAILURE;
  }

  printf("%g\n", simpleLogf(x));
  return 0;
}
