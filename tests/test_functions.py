import math
import subprocess
import mpmath
from os import path
import numpy as np

def correctLog(x):
  return np.log(np.float64(x))

def testLog(x):
  ret = subprocess.run([path.join((path.dirname(path.abspath(__file__))), path.pardir, "build/tests/logf"),
                        str(x)], capture_output=True, text=True, check=True)
  return float(ret.stdout.strip())

def getUlp(mpf):
  f = np.float32(mpf)
  next = np.nextafter(f, np.float32(np.inf))
  return (next - f).item()
