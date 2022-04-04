"""
Copyright © 2022 Naver Corporation. All rights reserved.
"""

import numpy as np
from scipy.linalg import orth


def is_ranking(ranking: np.ndarray, size: int = None):
    """
        Checks if `ranking` is a ranking, i.e. performs simple asserts to check if `ranking` is a valid ranking
    :param ranking: a ranking whose formatting is to be checked
    :type ranking: np.ndarray
    :param size: The size of the ranking, optional
    :type size: int
    """
    if size is not None:
        assert len(ranking) == size, "`ranking` must be of size " + str(size)
    else:
        size = len(ranking)
    assert np.all(np.sort(ranking) == np.arange(0, size)), "`ranking` is not a permutation of {0,...,n-1}"
    return True


def invert_permutation(permutation):
    """
    Inverts a permutation: If `permutation[i]==j`, then `invert_permutation(permutation)[j]==i`.

    :param permutation: A permutation represented as an array containing the integers 0 to n
    :type permutation: numpy.ndarray
    :return: The inverse permutation
    :rtype: numpy.ndarray
    """
    return np.argsort(permutation)


def project_on_subspace(point_to_project: np.ndarray, normal_vectors: np.ndarray) -> np.array:
    """
        Given an (m x n)-matrix `A`, this function computes the projection of `point_to_project` onto the linear subspace S = {x in R^n | Ax = 0}.

        The rows of matrix `A` are vectors orthogonal to the subspace
    :param point_to_project: The point to project on the linear subspace
    :type point_to_project: numpy.ndarray
    :param normal_vectors: A matrix whose rows are normal vectors to the subspace
    :return:
    """
    n = normal_vectors.shape[1]
    assert n == len(point_to_project), "The normal vectors must have the same dimension as the `point_to_project`"

    # if False:  # Method using span
    #     # Compute Moore-Penrose pseudo inverse
    #     Ad = A.T @ np.linalg.inv(A @ A.T)  # `Ad` is the Moore-Penrose pseudo inverse
    #     D = np.eye(n) - Ad @ A
    #     P = orth(D)
    #     return P @ (P.T @ point_to_project)
    # else:
    P = orth(normal_vectors.T)
    return point_to_project - P @ (P.T @ point_to_project)


def project_on_affine_subspace(point_to_project: np.ndarray, normal_vectors: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """
        Projects `point_to_project` onto the affine subspace with normal vectors `normal_vectors` and with an offset `offset`
    :param point_to_project: The point to project
    :param normal_vectors: A matrix whose rows contain the normal vectors to the subspace
    :param offset: Any point that is contained by the affine subspace
    :return: The projection of `point_to_project` onto the affine subspace
    """
    return project_on_subspace(point_to_project - offset, normal_vectors) + offset


def compute_unfairness(exposure_vector: np.ndarray, target: np.ndarray, p_norm: float = 2) -> float:
    """
        Computes an unfairness value of an exposure vector w.r.t a target exposure vector

    :param exposure_vector: The exposure vector of the documents
    :type exposure_vector: numpy.ndarray
    :param target: The vector of target exposures of the documents
    :type target: numpy.ndarray
    :param p_norm: The norm with which the unfairness is to be computed
    :type p_norm: float, optional
    :return: The unfairness of the exposure vector
    :rtype: float
    """
    assert 1 <= p_norm, "`p_norm` must be greater than 1"
    assert len(exposure_vector) == len(target), "`exposure_vector` and `target` must have same length"
    return np.linalg.norm(exposure_vector - target, ord=p_norm)


def nutility(exposure, pbm, relevance_vector) -> float:
    return exposure @ relevance_vector / (np.sort(relevance_vector) @ np.sort(pbm))


def vprint(msg: str, lvl: int = 0):
    if lvl == 1:
        print(msg)
