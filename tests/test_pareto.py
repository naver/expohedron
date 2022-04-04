"""
Copyright © 2022 Naver Corporation. All rights reserved.
"""

import unittest
from pareto import fairness_endpoint, pareto_curve_pbm
import numpy as np

# The functions to be tested need to be prefixed with `test_`, otherwise they are ignored


class Pareto(unittest.TestCase):
    def test_fairness_endpoint(self):
        # Case 1: 3 documents, 2 groups, no duplicate relevance values, endpoint is vertex
        gamma = np.array([3, 2, 1])
        tol = 1e-9
        rho = np.array([0.5, 0.6, 0.9])
        grouping = np.array([[1, 1, 0],
                             [0, 0, 1]])
        computed_point = fairness_endpoint(gamma, rho, grouping)
        theoretical_point = np.array([1., 3., 2.])
        self.assertTrue(np.all(np.abs(computed_point - theoretical_point) < tol))

        # Case 2: 4 documents, 2 groups, duplicate relevance values, endpoint is vertex
        gamma = np.array([4, 3, 2, 1])
        tol = 1e-9
        rho = np.array([0.5, 0.6, 0.9, 0.9])
        grouping = np.array([[0, 1, 0, 1],
                             [1, 0, 1, 0]])
        computed_point = fairness_endpoint(gamma, rho, grouping)
        theoretical_point = np.array([1, 2, 4, 3])
        self.assertTrue(np.all(np.abs(computed_point - theoretical_point) < tol))

        # Case 3: 10 documents, 10 groups, duplicate relevance values, endpoint is barycenter
        gamma = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        tol = 1e-9
        rho = np.array([0.5, 0.6, 0.9, 0.8, 0.1, 0.3, 0.7, 0.2, 0.4, 1])
        grouping = np.eye(10)
        computed_point = fairness_endpoint(gamma, rho, grouping)
        theoretical_point = 5.5 * np.ones(10)
        self.assertTrue(np.all(np.abs(computed_point - theoretical_point) < tol))

        # Case 4: 4 documents, 2 groups, no duplicate relevance values, endpoint is not vertex
        gamma = np.array([4, 3, 2, 1])
        tol = 1e-9
        rho = np.array([0.5, 0.6, 0.8, 0.9])
        grouping = np.array([[0, 1, 0, 1],
                             [1, 0, 1, 0]])
        computed_point = fairness_endpoint(gamma, rho, grouping)
        theoretical_point = np.array([1.5, 1.5, 3.5, 3.5])
        self.assertTrue(np.all(np.abs(computed_point - theoretical_point) < tol))

        # Case 4: Meritocratic individual fairness
        ndoc = 10
        gamma = np.random.rand(ndoc)
        rho = np.random.rand(ndoc)
        computed_point = fairness_endpoint(gamma, rho, fairness="meritocratic", grouping=None)
        theoretical_point = rho / np.sum(rho) * np.sum(gamma)
        self.assertTrue(np.all(np.abs(computed_point - theoretical_point) < tol))

    def test_pareto_curve_pbm(self):
        # Case 1
        gamma = np.array([4, 3, 2, 1])
        tol = 1e-9
        rho = np.array([0.5, 0.6, 0.8, 0.9])
        grouping = np.array([[0, 1, 0, 1],
                             [1, 0, 1, 0]])
        computed_pareto_curve = pareto_curve_pbm(gamma, rho, fairness="demographic", grouping=grouping)
        theoretical_pareto_curve = [np.array([1.5, 1.5, 3.5, 3.5]), np.array([1, 2, 3, 4])]
        self.assertTrue(np.all(np.abs(theoretical_pareto_curve[0] - computed_pareto_curve[0]) < tol))
        self.assertTrue(np.all(np.abs(theoretical_pareto_curve[1] - computed_pareto_curve[1]) < tol))

        # Case 2: Meritocratic individual fairness, 3 documents, regular hexagon
        gamma = np.array([1, 2, 3])
        tol = 1e-9
        rho = np.array([2, 1.5, 2.5])
        computed_pareto_curve = pareto_curve_pbm(gamma, rho, fairness="meritocratic", grouping=None)
        theoretical_pareto_curve = [np.array([2, 1.5, 2.5])]


if __name__ == '__main__':
    unittest.main()
