'''
Testing motion backend.
'''

import unittest
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from motion import *

_NUM_TRIALS_ = 1000

class test_motion(unittest.TestCase):
  '''
  Test functions in motion.py
  '''
  def test_derivs_on_poly(self):
    '''
    Test computation of derivatives for polynomials.
    '''
    for seed in range(_NUM_TRIALS_):
      np.random.seed(seed)

      deg = np.random.randint(low = 0, high = 6)

      coeffs = np.random.uniform(low = -10, high = 10, size = (deg + 1))

      s = Polynomial(coeffs)

      t_min = np.random.uniform(low = -10, high = 10)
      t_max = t_min + np.random.uniform(low = 0.01, high = 5)
      num_ts = np.random.randint(low = 1000, high = 5000)

      ts, psns, vels, accs = compute_psn_vel_acc(s, t_min, t_max, num_ts = num_ts)

      v = s.deriv()
      a = v.deriv()

      psns_true = s(ts)
      vels_true = v(ts)
      accs_true = a(ts)
      self.assertTrue(np.allclose(psns_true, psns))
      self.assertTrue(np.allclose(vels_true, vels, rtol = 0.01, atol = 0.01))
      self.assertTrue(np.allclose(accs_true, accs, rtol = 0.01, atol = 0.01))


  def test_derivs_on_sincos(self):
    '''
    Test computation of derivatives on p(t) = c_1\sin(c_2t) + c\_3cos(c_4t). Note v(t) = c_1c_2\cos(c_2t) 
      - c_3c_4\sin(c_4t) and a(t) = -c_1c_2^2\sin(c_2t) - c3c_4^2\cos(c_4t)
    '''
    for seed in range(_NUM_TRIALS_):
      np.random.seed(seed)

      c1 = np.random.uniform(low = -10, high = 10)
      c2 = np.random.uniform(low = -10, high = 10)
      c3 = np.random.uniform(low = -10, high = 10)
      c4 = np.random.uniform(low = -10, high = 10)

      s = lambda t: c1 * np.sin(c2 * t) + c3 * np.cos(c4 * t)
      v = lambda t: c1 * c2 * np.cos(c2 * t) - c3 * c4 * np.sin(c4 * t)
      a = lambda t: -c1 * c2**2 * np.sin(c2 * t) - c3 * c4**2 * np.cos(c4 * t)


      t_min = np.random.uniform(low = -10, high = 10)
      t_max = t_min + np.random.uniform(low = 0.01, high = 5)
      num_ts = np.random.randint(low = 1000, high = 5000)

      ts, psns, vels, accs = compute_psn_vel_acc(s, t_min, t_max, num_ts = num_ts)

      psns_true = s(ts)
      vels_true = v(ts)
      accs_true = a(ts)

      self.assertTrue(np.allclose(psns_true, psns))
      self.assertTrue(np.allclose(vels_true, vels, rtol = 0.02, atol = 0.02))
      self.assertTrue(np.allclose(accs_true, accs, rtol = 0.02, atol = 0.02))

  
if __name__ == "__main__":
  unittest.main()   