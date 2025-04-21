#include <stdio.h>
#include <stdlib.h>

#include "lalogf/logf.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: ./a.out <floating-point number>");
  }

  float x = strtof(argv[1], NULL);
  printf("%.10f\n", lalogf(x));
  return 0;
}
