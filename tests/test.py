import math
import subprocess
import mpmath
from os import path
import numpy as np
import unittest
from test_functions import testLog, getUlp, correctLog

MAX_ULP_ERROR = 3.5

class TestLogf(unittest.TestCase):
  def test_ulp_precision(self):
    segments = [
        np.linspace(2.0**e, 2.0**(e + 1), 20, dtype=np.float32)
        for e in range(-126, 127)
    ]
    test_set = np.concatenate(segments)
    ulp_errors = []
    for x in test_set:
      expected = correctLog(x)
      res = testLog(x)
      ulp = getUlp(res)
      error = abs(mpmath.mpf(res) - expected)
      ulp_error = float(abs(error / mpmath.mpf(ulp))) if ulp != 0.0 else 0.0
      print(f'x = {x}, ulp_error = {ulp_error}')
      self.assertTrue(ulp_error <= MAX_ULP_ERROR)
      ulp_errors.append(ulp_error)

    max_ulp_error = max(ulp_errors)
    print(f'max ulp error: {max_ulp_error}')

if __name__ == "__main__":
  unittest.main()
