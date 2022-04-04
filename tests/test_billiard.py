"""
Copyright © 2022 Naver Corporation. All rights reserved.
"""

import unittest
from billiard import *


class Pareto(unittest.TestCase):
    def test_billiard(self):
        gen = billiard_word([2/5, 3/5])
        self.assertTrue([next(gen) for i in range(10)] == [0, 1, 1, 0, 1]*2)

        gen = billiard_word([3/5, 2/5])
        self.assertTrue([next(gen) for i in range(10)] == [0, 1, 0, 1, 0]*2)

        gen = billiard_word([1/3, 1/3, 1/3])
        self.assertTrue([next(gen) for i in range(30)] == [0, 1, 2]*10)

        gen = billiard_word([3/5, 1/10, 3/10])
        self.assertTrue([next(gen) for i in range(30)] == [0, 1, 2, 0, 0, 2, 0, 0, 2, 0]*3)


if __name__ == '__main__':
    unittest.main()
