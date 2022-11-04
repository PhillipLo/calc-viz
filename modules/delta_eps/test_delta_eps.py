'''
Testing delta_eps backend.
'''

import unittest
import numpy as np
from delta_eps import *

_NUM_TRIALS_ = 100

class test_delta_eps(unittest.TestCase):
  '''
  Test functions in delta_eps.py
  '''
  def test_compute_delta(self):
    '''
    Test computation of delta from epsilon for linear functions. If f(x) = mx + b, then given
      eps, delta should equal eps / m
    '''
    for seed in range(_NUM_TRIALS_):
      np.random.seed(seed)
      
      m = np.random.uniform(-10, 10)
      b = np.random.uniform(-10, 10)

      f = lambda x: m * x + b 

      x_minmax = np.random.uniform(-10, 10, size = 2)
      x_min = np.min(x_minmax)
      x_max = np.max(x_minmax)

      y_range = np.abs(f(x_max) - f(x_min)) # max(f) - min(f)
      eps = np.random.uniform(1e-3, 1e-1) * y_range

      delta_true = eps / np.abs(m)

      # so that (a - delta, a + delta) stays in domain
      a = np.random.uniform(x_min + delta_true, x_max - delta_true)
      L = f(a)

      delta_stepsize = np.random.uniform(1e-7, 1e-4)

      delta = compute_delta(f, a, L, eps, x_min, x_max, delta_stepsize = delta_stepsize)

      self.assertTrue(delta_true - delta_stepsize <= delta)
      self.assertTrue(delta <= delta_true + delta_stepsize)


if __name__ == "__main__":
  unittest.main()   



