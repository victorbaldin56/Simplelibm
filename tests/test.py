import math
import subprocess
import mpmath
from os import path
import numpy as np

def correctLog(x):
  return mpmath.log(x)

def testLog(x):
  ret = subprocess.run(path.join((path.abspath(__file__)), path.pardir, "build/logf"),
                       str(x), capture_output=True, text=True, check=True)
  return float(ret.stdout.strip())

def getUlp(mpf):
  f = np.float32(mpf)
  next = np.nextafter(f, np.float32(np.inf))
  return (next - f).item()
