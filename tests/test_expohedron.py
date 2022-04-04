"""
Copyright © 2022 Naver Corporation. All rights reserved.
"""

import unittest
from expohedron import *
import numpy as np


# The functions to be tested need to be prefixed with `test_`, otherwise they are ignored


class Expohedron(unittest.TestCase):
    def test_majorized(self):
        gamma = np.array([1, 2, 3, 4, 5])
        # Case 1: Properly majorized
        self.assertTrue(majorized(np.array([4, 4, 2, 2, 3]), gamma))
        # Case 2: Majorized at the limit
        self.assertTrue(majorized(np.array([5, 4, 3, 2, 1]), gamma))
        # Case 3: Not majorized but cumsum equal
        self.assertFalse(majorized(np.array([5, 5, 1, 1, 3]), gamma))
        # Case 4: Not majorized
        self.assertFalse(majorized(np.array([5, 1, 2, 2, 1]), gamma))

    def test_error_correction(self):
        tol = 1e-15

        # Case 1: `point` is a vertex
        gamma = np.array([1, 2, 3, 4])
        n = len(gamma)
        point = np.array([2, 1, 4, 3])
        face = Face(gamma, np.array([1, 0, 3, 2]), np.array([0, 1, 2, 3]))
        assert face.contains(point)
        noise = 1e-9
        noisy_point = (1 - noise) * point + noise * np.ones(n) * np.mean(gamma)
        corrected_point = error_correction(noisy_point, point - np.ones(n) * np.mean(gamma), face, tol=tol)
        self.assertTrue(np.all(np.abs(corrected_point - point) < tol))

        # Case 2:
        gamma = np.array([1, 2, 3, 4])
        n = len(gamma)
        point = np.array([2, 1, 4, 3])
        face = Face(gamma, np.array([1, 0, 3, 2]), np.array([0, 1, 2, 3]))
        assert face.contains(point)
        noise = 1e-9
        noisy_point = (1 - noise) * point + noise * np.ones(n) * np.mean(gamma)
        corrected_point = error_correction(noisy_point, point - np.ones(n) * np.mean(gamma), face, tol=tol)
        self.assertTrue(np.all(np.abs(corrected_point - point) < tol))

    def test_Face(self):
        """Test the function of the `Face` object"""
        # Case 1: `contains`
        gamma = np.array([1, 2, 3, 4])
        face = Face(gamma, np.array([0, 1, 2, 3]), np.array([1, 3]))
        self.assertTrue(face.contains(np.array([1.1, 1.9, 3.6, 3.4])))
        self.assertTrue(face.contains(np.array([1, 2, 4, 3])))

    def test_identify_face(self):
        gamma = np.array([1, 2, 3])

        # Case 1: Edge
        point = np.array([2.5, 2.5, 1])
        face = identify_face(gamma, point)
        self.assertEqual(face.dim, 1)  # Check dimension
        self.assertTrue(np.all(face.zone == np.array([2, 0, 1])) or np.all(face.zone == np.array([2, 1, 0])))  # Check zone
        self.assertTrue(np.all(face.splits == np.array([0, 2])))  # Check splits

        # Case 2: Vertex
        point = np.array([2, 3, 1])
        face = identify_face(gamma, point)
        self.assertEqual(face.dim, 0)  # Check dimension
        self.assertTrue(np.all(face.zone == np.array([2, 0, 1])))  # Check zone
        self.assertTrue(np.all(face.splits == np.array([0, 1, 2])))  # Check splits

        # Case 2: Facet
        point = np.array([1.5, 2, 2.5])
        face = identify_face(gamma, point)
        self.assertEqual(face.dim, 2)  # Check dimension
        self.assertTrue(np.all(face.splits == np.array([2])))  # Check splits

    def test_find_face_intersection_same_ordering(self):
        gamma = np.array([1, 2, 3])
        tol = 1e-12

        # Case 1: From barycenter
        point = np.array([2, 2, 2])
        direction = np.array([-1.5, 0.5, 1])
        u = find_face_intersection_same_ordering(gamma, point, direction)
        self.assertTrue(np.all(np.abs(u - np.array([1, 2 + 1 / 3, 2 + 2 / 3]) < tol)))

        # Case 2: Intersection within face
        point = np.array([1.6, 1.4, 3])
        direction = np.array([1, -1, 0])
        u = find_face_intersection_same_ordering(gamma, point, direction, override_order_constraint=True)
        self.assertTrue(np.all(np.abs(u - np.array([2., 1., 3.]) < tol)))

        # Case 2: Intersection within face, specified zone
        point = np.array([1.5, 1.5, 3])
        direction = np.array([1, -1, 0])
        u = find_face_intersection_same_ordering(gamma, point, direction, override_order_constraint=True, zone=np.array([1, 0, 2]))
        self.assertTrue(np.all(np.abs(u - np.array([2., 1., 3.]) < tol)))

        # Case 4: 10 dimensions
        gamma = np.arange(0, 10) + 1
        point = np.mean(gamma) * np.ones(10)
        direction = np.array([1, -1, 0, 0, 0, 0, 0, 0, 0, 0])
        u = find_face_intersection_same_ordering(gamma, point, direction, zone=np.argsort(direction))
        self.assertTrue(np.all(np.abs(u - np.array([10, 1, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]) < tol)))

    def test_find_face_intersection(self):
        gamma = np.array([1, 2, 3])
        tol = 1e-12

        # Case 1: From barycenter
        point = np.array([2, 2, 2])
        direction = np.array([-1.5, 0.5, 1])
        u = find_face_intersection(gamma, point, direction)
        self.assertTrue(np.all(np.abs(u - np.array([1, 2 + 1 / 3, 2 + 2 / 3]) < tol)))

        # Case 2: Intersection not in same zone as starting point
        point = np.array([1.5, 2, 2.5])
        direction = np.array([1, 0, -1])
        u = find_face_intersection(gamma, point, direction)
        self.assertTrue(np.all(np.abs(u - np.array([3., 2., 1.]) < tol)))

        # Case 3: Intersection within face
        point = np.array([1.6, 1.4, 3])
        direction = np.array([1, -1, 0])
        u = find_face_intersection(gamma, point, direction)
        self.assertTrue(np.all(np.abs(u - np.array([2., 1., 3.]) < tol)))

        # Case 4: 10 dimensions
        gamma = np.arange(0, 10) + 1
        point = np.mean(gamma) * np.ones(10)
        direction = np.array([1, -1, 0, 0, 0, 0, 0, 0, 0, 0])
        u = find_face_intersection(gamma, point, direction)
        self.assertTrue(np.all(np.abs(u - np.array([10, 1, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]) < tol)))

        # Case 5: Inadmissible direction
        gamma = np.array([1, 2, 3])
        point = np.array([1.5, 2, 2.5])
        direction = np.array([1, 1, -1])
        self.assertRaises(AssertionError, lambda: find_face_intersection(gamma, point, direction))

    def test_find_face_subspace(self):
        gamma = np.array([1, 2, 3])
        tol = 1e-12

        # Case 1: Facet
        face = Face(gamma, np.array([0, 1, 2]), splits=np.array([2]))
        face_orth = find_face_subspace(face)
        A = np.ones(3) / np.linalg.norm(np.ones(3))
        self.assertTrue(np.all(np.abs(face_orth - A) < tol))

        # Case 2: Edge
        face = Face(gamma, np.array([0, 2, 1]), np.array([0, 2]))
        face_orth = find_face_subspace(face)
        self.assertTrue(np.all(np.abs(face_orth.T @ (gamma[np.array([0, 2, 1])] - gamma[np.array([0, 1, 2])])) < tol))

        # Case 3: Vertex
        face = Face(gamma, np.array([0, 2, 1]), np.array([0, 1, 2]))
        face_orth = find_face_subspace(face)
        self.assertEqual(np.linalg.matrix_rank(face_orth), 3)

        # Case 4: 10 documents, 3 splits
        gamma = np.arange(0, 10) + 1
        order = np.arange(0, 10)
        face = Face(gamma, order, np.array([1, 7, 9]))  # `gamma` is in `face`
        face_orth = find_face_subspace(face)
        self.assertTrue(np.all(np.abs(face_orth.T @ (np.array([1.1, 1.9, 3, 4, 5, 5, 8, 8, 9.5, 9.5]) - gamma)) < tol))

        # Case 5: Check persistent bug
        point = np.array([0.43082352, 0.86915565, 0.42050336, 0.90150568, 0.56522123, 0.64855314, 0.75271679, 0.73036695, 0.58062715, 0.65549705])
        n_doc = 10
        gamma = 1 / np.log(np.arange(0, n_doc) + 2)  # The DCG exposures
        np.random.seed(5)
        rho = np.random.rand(n_doc)
        face = Face(gamma, np.array([2, 0, 4, 8, 5, 9, 7, 6, 1, 3]), np.array([1, 9]))
        face_orth = find_face_subspace(face)
        self.assertEqual(np.linalg.matrix_rank(face_orth), 2)
        direction_1 = np.array([-1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        self.assertTrue(np.all(np.abs(face_orth.T @ direction_1) < tol))

        direction = project_on_subspace(rho, face_orth.T)

    def test_caratheodory_decomposition_pbm_gls(self):
        gamma = np.array([1, 2, 3, 4, 5])
        tol = 1e-9

        # Case 1: Arbitrary point
        point = np.array([3., 3., 2., 4., 3.])
        self.assertTrue(majorized(point, gamma))
        coefficients, vertices = caratheodory_decomposition_pbm_gls(gamma, point)
        reconstructed_point = vertices @ coefficients
        self.assertTrue(np.all(np.abs(point - reconstructed_point) < tol))

        # Case 2: Barycenter
        point = np.array([3., 3., 3., 3., 3.])
        self.assertTrue(majorized(point, gamma))
        coefficients, vertices = caratheodory_decomposition_pbm_gls(gamma, point)
        reconstructed_point = vertices @ coefficients
        self.assertTrue(np.all(np.abs(point - reconstructed_point) < tol))

        # Case 3: Point in face
        point = np.array([1., 2.5, 2.5, 4.5, 4.5])
        self.assertTrue(majorized(point, gamma))
        coefficients, vertices = caratheodory_decomposition_pbm_gls(gamma, point)
        reconstructed_point = vertices @ coefficients
        self.assertTrue(np.all(np.abs(point - reconstructed_point) < tol))

        # Case 4: DCG expohedron
        gamma = 1 / np.log(np.arange(0, 3) + 2)
        n = len(gamma)
        point = np.array([0.3, 0.5, 0.55])
        point = point / np.sum(point) * np.sum(gamma)
        barycenter = np.ones(n) * np.mean(gamma)
        point = find_face_intersection(gamma, barycenter, point - barycenter)
        assert identify_face(gamma, point).dim < n - 1, "The intersection must lie on a face of the expohedron that is not the expohedron itself"
        coefficients, vertices = caratheodory_decomposition_pbm_gls(gamma, point)
        reconstructed_point = vertices @ coefficients
        self.assertTrue(np.all(np.abs(point - reconstructed_point) < tol))

    def test_post_correction(self):
        # Case 1: 4 documents
        gamma = np.array([1, 2, 3])
        tol = 1e-15
        face = Face(gamma, np.array([0, 1, 2]), np.array([1, 2]))
        point = np.array([2, 1, 3])
        post_point = post_correction(face, point)
        self.assertTrue(np.all(np.abs(point - post_point) < tol))

        # Case 2: 4 documents
        gamma = np.array([1, 2, 3, 4])
        tol = 1e-15
        face = Face(gamma, np.array([0, 1, 2, 3]), np.array([1, 3]))
        point = np.array([1.1, 1.9, 3.6, 3.4])
        post_point = post_correction(face, point + 0.01 * np.array([0.1, 0.1, 0.2, 0.2]))
        self.assertTrue(np.all(np.abs(point - post_point) < tol))

    def test_project_on_face_subspace(self):
        # Case 1
        # todo
        pass


if __name__ == '__main__':
    unittest.main()
