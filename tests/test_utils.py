"""
Copyright © 2022 Naver Corporation. All rights reserved.
"""

import unittest
from utils import *

# The functions to be tested need to be prefixed with `test_`, otherwise they are ignored


class Utils(unittest.TestCase):
    def test_invert_permutation(self):
        pi = np.array([1, 0, 5, 3, 2, 4])
        inv_pi = invert_permutation(pi)
        self.assertTrue(np.all(inv_pi == np.array([1, 0, 4, 3, 5, 2])))

    def test_project_on_subspace(self):
        G = np.array([[1, 0, 0, 1, 0, 0],
                      [0, 1, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1]])
        z = np.random.rand(6)
        y = project_on_subspace(z, G)
        self.assertLess(np.abs(np.sum(G @ y)), 1e-12)


if __name__ == '__main__':
    unittest.main()
