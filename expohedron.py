"""
    Copyright © 2022 Naver Corporation. All rights reserved.

    This file regroups a number of function about the expohedron
"""

from scipy.special import comb
import warnings
import pareto as pareto

from utils import *

ULTRA_LOW_TOLERANCE = 5e-3
LOW_TOLERANCE = 1e-6
DEFAULT_TOLERANCE = 1e-9
HIGH_TOLERANCE = 1e-15
MAX_TOLERANCE = np.finfo(np.float).eps  # 2.220446049250313e-16


class PBMexpohedron:
    relevance_vector: np.ndarray
    pbm: np.ndarray
    n: int
    prp_vertex: np.ndarray
    prp_utility: float
    prp_unfairness: float

    def __init__(self, pbm: np.ndarray, relevance_vector: np.ndarray):
        assert np.all(0 <= relevance_vector) and np.all(relevance_vector <= 1), "The values of `relevance_vector` must all be between 0 and 1"
        self.relevance_vector = relevance_vector
        self.pbm = pbm
        self.n = len(relevance_vector)
        self.prp_vertex = pbm[invert_permutation(np.argsort(-relevance_vector))]
        self.prp_utility = self.prp_vertex @ relevance_vector

    def __repr__(self):
        string1 = "PBM expohedron:\n\trelevance_vector of length " + str(self.n) + " :\n\t" + str(self.relevance_vector)
        string2 = "\n\tPBM = " + str(self.pbm)
        return string1 + string2

    def __str__(self):
        return self.__repr__()

    def is_inside(self, point: np.ndarray) -> bool:
        return majorized(point, self.pbm)

    def get_vertex(self, ranking: np.ndarray) -> np.ndarray:
        """
            Given a ranking, a relevance vector and an abandon probability, computes the exposure vector (in the document space)

            The ranking is such that when applied to the document indices
        :param ranking: A matrix of size D x D
        :type ranking: numpy.array
        :return: A column vector of size D
        :rtype: numpy.array
        """
        assert is_ranking(ranking)
        return self.pbm[invert_permutation(ranking)]

    def utility(self, exposure_vector: np.ndarray) -> float:
        return exposure_vector @ self.relevance_vector

    def nutility(self, exposure_vector: np.ndarray) -> float:
        return self.utility(exposure_vector) / self.prp_utility

    def unfairness(self, exposure_vector: np.ndarray, fairness: str = "meritocratic",
                   p_norm: float = 2, meritocratic_endpoint: str = "intersection") -> float:
        """

        :param fairness:
        :param p_norm:
        :param meritocratic_endpoint:
        :return:
        """
        if fairness == "demographic":
            target = self.demographic_fairness_target()
        elif fairness == "meritocratic":
            target = self.meritocratic_target_exposure(type=meritocratic_endpoint)
        else:
            raise ValueError("Invalid value for `fairness`")
        return compute_unfairness(exposure_vector, target, p_norm)

    def nunfairness(self, exposure_vector: np.ndarray, fairness: str = "meritocratic",
                    p_norm: float = 2, meritocratic_endpoint: str = "intersection") -> float:
        """
            Normalized unfairness
        :param fairness:
        :param p_norm:
        :param meritocratic_endpoint:
        :return:
        """
        return self.unfairness(exposure_vector, fairness, p_norm, meritocratic_endpoint) / \
               self.unfairness(self.prp_vertex, fairness, p_norm, meritocratic_endpoint)

    def demographic_fairness_target(self) -> np.ndarray:
        """
            Computes the feasible demographic target exposure vector

        :param expohedron: The expohedron to consider
        :type expohedron: DBNexpohedron
        :return: The demographic target exposure
        :rtype: numpy.ndarray
        """
        return np.ones(self.n) * (self.prp_vertex @ np.ones(self.n)) / (np.ones(self.n) @ np.ones(self.n))

    def meritocratic_target_exposure(self, type: str="intersection") -> np.ndarray:
        """
                Computes the feasible meritocratic target exposure vector

            :param type: How to find a feasible target if the "true" one is infeasible
            :type type: str, optional
            :return: The meritocratic target exposure
            :rtype: numpy.ndarray
        """
        assert type == "intersection", "Only intersection method is currently supported"
        true_fairness_endpoint = self.relevance_vector * (self.prp_vertex @ np.ones(self.n)) / (self.relevance_vector @ np.ones(self.n))
        if self.is_inside(true_fairness_endpoint):
            return true_fairness_endpoint
        else:
            demographic_fairness_point = self.demographic_fairness_target()
            direction = true_fairness_endpoint - demographic_fairness_point
            return find_face_intersection(self.pbm, demographic_fairness_point, direction)

    def target_exposure(self, fairness: str, meritocratic_endpoint: str = "intersection"):
        if fairness == "demographic":
            return self.demographic_fairness_target()
        elif fairness == "meritocratic":
            return self.meritocratic_target_exposure(type=meritocratic_endpoint)
        else:
            raise ValueError("Invalid value for `fairness`")


def majorized(a: np.array, b: np.array, tolerance: float = LOW_TOLERANCE) -> bool:
    """
        Checks whether `a` is majorized by `b`: a<b

        :param a: The left hand side of the comparison a<b
        :type a: numpy.array
        :param b: The right hand side of the comparison a<b
        :type b: numpy.array
        :param tolerance: the tolerance that is allowed
        :type tolerance: float
        :return: `True` if a < b, false otherwise
        :rtype: bool
    """
    return np.all(np.cumsum(-np.sort(-a)) <= np.cumsum(-np.sort(-b)) + tolerance) and np.abs(np.sum(a) - np.sum(b)) < tolerance


def sample_point_in_expohedron(gamma: np.ndarray, size: int = 1) -> np.ndarray:
    """
    Sample a random point inside the expohedron.

    This is achieved by accept-reject sampling in the simplex
    :param gamma: The PBM exposures
    :type gamma: numpy.ndarray
    :param size: The number of points to sample
    :type size: int, optional
    :return: A matrix whose rows are the sampled points in the expohedron
    :rtype: numpy.ndarray
    """
    n = len(gamma)
    result = np.zeros((size, n)) * np.nan
    k = 0
    while k < size:
        sample = np.random.uniform(low=0, high=1, size=n)
        sample = sample / np.sum(sample) * np.sum(gamma)
        if majorized(sample, gamma):
            result[k, :] = sample
            k += 1
    return result


class Face:
    """
        Implements a face object of an expohedron.

        The face are all points in space such that
    """
    gamma: np.ndarray
    zone: np.ndarray
    splits: np.ndarray
    dim: int

    def __init__(self, gamma, zone, splits):
        n = len(gamma)
        assert np.all(np.sort(zone) == np.arange(0, n)), "zone must be a permutation"
        assert n == len(zone)
        assert issubclass(splits.dtype.type, np.integer), "splits must contain integer indices"
        assert np.all(0 <= splits) and np.all(splits < n), "The indices in split must be in the adequate range"
        assert len(splits) > 0, "All faces must have at least one split. The whole expohedron has exactly one split"

        self.gamma = np.sort(gamma)
        self.zone = zone
        self.splits = splits
        self.dim = n - len(splits)

    def contains(self, point: np.array, tolerance: float = 1e-12) -> bool:
        """
            Checks if a point is inside the face
        :param point:
        :return:
        :rtype: bool
        """
        maj = majorized(point, self.gamma)  # majorization condition
        face_condition: bool = len(np.setdiff1d(self.splits, np.where(np.abs(np.cumsum(point[self.zone]) - np.cumsum(self.gamma)) < tolerance)[0])) == 0  # Check if the splits
        # of `point` are a subset of `self.splits`
        return maj and face_condition

    def equal(self, face: "Face") -> bool:
        """
            Checks if `self` is equal to `face`
        :param face:
        :return:
        :rtype: bool
        """
        return np.all(invert_permutation(self.zone)[self.splits] == invert_permutation(face.zone)[face.splits])


def error_correction(point: np.ndarray, direction: np.ndarray, face: Face, tol: float = HIGH_TOLERANCE) -> np.ndarray:
    """
        Correct numerical imprecisions in a point on a face

        Given a point `point` on a face `face` of an expohedron, and a half-line `direction` on which `point` lies`, corrects numerical imprecisions in `point`.
        NB: The imprecise point MUST still be majorized by `face.gamma`.
    :param point: The point to correct
    :type point: numpy.ndarray
    :param direction: The direction in which the point must be corrected
    :type direction: numpy.ndarray
    :param face: The face on which the point actually lies
    :type face: Face
    :param tol: The allowed tolerance for the corrected point
    :type tol: float, optional
    :return: The corrected point
    :rtype: numpy.ndarray
    """
    n = len(point)
    assert n == len(direction), "`direction must have same length as `point`"
    assert majorized(point, face.gamma), "`point` must be majorized by `face.gamma`"

    zone = np.argsort(direction)
    # `face.zone` is the base we are working in
    Gk = np.cumsum(face.gamma[zone])
    Sk = np.cumsum(point[zone])
    Dk = np.cumsum(direction[zone])
    Lambda = min(np.abs((Gk - Sk)[0:(n - 1)] / Dk[0:(n - 1)]))
    return point + Lambda * direction
    # todo check if this works whatever zone `face` is defined in
    # Answer: It does not


def project_on_face_subspace(point_to_project: np.ndarray, face: Face, affine: bool = False):
    """
        Projects  the point `point_to_project` onto the linear subspace corresponding to `face`. If `affine is True` then the affine subspace is used
    :param point_to_project: The point to project
    :type point_to_project: numpy.ndarray
    :param face: The face on whose subspace we seek to project
    :type face: Face
    :param affine: Whether the projection should be on the affine subspace. Default is `False`
    :type affine: bool, optional
    :return: The projection
    :rtype: numpy.ndarray
    """
    n = len(point_to_project)
    assert n == len(face.gamma), "`point_to_project` must have same length as `face.gamma`."
    face_orth = find_face_subspace(face)
    if affine:
        return project_on_affine_subspace(point_to_project, face_orth.T, offset=face.gamma[face.zone])  # todo check this
    else:
        return project_on_subspace(point_to_project, face_orth.T)


def find_face_subspace(face: Face) -> np.ndarray:
    """
        Given a face of the expohedron, finds all the normal vectors to the smallest linear subspace containing that face.

    :param face: The face whose subspace is to be found
    :type face: Face
    :return: A matrix whose rows are an orthonormal basis to the complement of the linear subspace. A such that S = {x | Ax=0}.
    :rtype: numpy.ndarray
    """
    return find_face_subspace_without_parent(face)


def find_face_subspace_without_parent(face: Face) -> np.ndarray:
    """
        Computes the smallest linear subspace in which `face` lies. Returns the orthonormal vectors to the subspace.
    :param face: The face whose subspace is to be computed
    :type face: Face
    :return: The normal vectors
    """
    n = len(face.gamma)  # The dimensionality of the space
    n_orth = n - face.dim  # The dimensionality of the orthogonal space
    A = np.zeros((n_orth, n))
    for j in np.arange(0, n_orth):
        i = face.splits[j]
        nu = np.ones(n)
        s1 = np.sum(face.gamma[0:i+1])
        s2 = np.sum(face.gamma[i+1:n])
        if s2 == 0:
            psi = 1
        else:
            psi = s1 / s2
        nu[i+1:n] = -psi
        A[j, :] = nu[invert_permutation(face.zone)]
    return orth(A.T)


def find_face_intersection(gamma: np.ndarray, starting_point: np.ndarray, direction: np.ndarray, precision: float = 1e-12) -> np.ndarray:
    """
        Finds the intersection of a half-line with the border of the expohedron

        Given a starting point `starting_point` in an expohedron given by `gamma` and given a direction vector `direction`, this function find
        the intersection of the half line starting at `starting_point` with direction `direction` with the border of the polytope
    :param gamma: A vertex of the PBM-expohedron
    :type gamma: numpy.ndarray
    :param starting_point: the starting point of the half line
    :type starting_point: numpy.ndarray
    :param direction: the direction of the half line
    :type direction: numpy.ndarray
    :param precision: THe precision to be used for the bisection method
    :type precision: float, optional
    :return: The intersection of the half-line with the border of the polytope
    :rtype: numpy.ndarray
    """
    if np.linalg.norm(direction) < precision:
        Warning("`direction` has norm lower than " + str(precision) + ". Assumed to be 0. Returning `starting_point`.")
        return starting_point
    assert np.sum(direction) < LOW_TOLERANCE, "`direction`'s elements must sum to zero"

    if np.all(np.argsort(starting_point) == np.argsort(direction)):
        return find_face_intersection_same_ordering(gamma, starting_point, direction)
    else:
        return find_face_intersection_bisection(gamma, starting_point, direction, precision)


def find_face_intersection_same_ordering(gamma: np.ndarray, starting_point: np.ndarray, direction: np.ndarray, override_order_constraint: bool = False,
                                         zone: np.ndarray = None) -> np.ndarray:
    """
        Given a starting point `starting_point` in an expohedron given by `gamma` and given a direction vector `direction`, this function finds

        the intersection of the half line starting at `starting_point` with direction `direction` with the border of the polytope
        This is done under the assumption that all points in the half-line have the same ordering
        voir p. 71 et 88 de mon carnet
    :param gamma: A vertex of the PBM-expohedron
    :type gamma: numpy.ndarray
    :param starting_point: the starting point of the half line
    :type starting_point: numpy.ndarray
    :param direction: the direction of the half line
    :type direction: numpy.ndarray
    :param override_order_constraint: If `True` the order constraint is not checked and can be overridden
    :type override_order_constraint: bool, optional
    :param zone: This parameter serves to lift ambiguity whenever the starting point is in several zones. If this argument is given and `starting_point` is indeed in `zone`, then `zone is chosen`
    :type zone: np.ndarray, optional
    :return: The intersection of the half-line with the border of the polytope
    :rtype: numpy.ndarray
    """
    if zone is not None:
        assert np.all(starting_point[zone] == np.sort(starting_point)), "The starting point is not in given zone"
    else:
        zone = np.argsort(starting_point)
    if not override_order_constraint:
        assert np.all(np.sort(starting_point) == starting_point[np.argsort(direction)]), "Both starting point and direction need to have the same ordering"
    assert len(gamma) == len(starting_point), "gamma and starting_point must have same length"
    assert len(gamma) == len(direction), "gamma and direction must have same length"
    assert np.sum(direction) < LOW_TOLERANCE, "`direction`'s elements must sum to zero"

    n = len(gamma)
    Gk = np.cumsum(np.sort(gamma))
    Sk = np.cumsum(starting_point[zone])
    Dk = np.cumsum(direction[zone])
    # eliminate the coordinates where Dk is zero; no information about Lambda can be obtained from them. todo: refer to a proof in paper

    # Lambda = min((Gk - Sk)[0:(n - 1)] / Dk[0:(n - 1)])
    indices = np.where(np.abs(Dk) > 1e-12)
    bounds = (Gk - Sk)[indices] / Dk[indices]
    Lambda = min(bounds[np.where(bounds >= 0)], default=0)

    return starting_point + Lambda * direction


def find_face_intersection_bisection(gamma: np.ndarray, starting_point: np.ndarray, direction: np.ndarray, precision: float) -> np.ndarray:
    """
        Executes a bisection search in the PBM-expohedron using the majorization criterion.

        It finds the intersection of a half-line starting at `starting_point` in the direction `direction` with the border of the expohedron defined by `gamma`.
    :param gamma: Any vertex of the PBM-expohedron
    :type gamma: numpy.ndarray
    :param starting_point: The starting point of the half-line
    :type starting_point: numpy.ndarray
    :param direction: The direction of the half-line
    :type direction: numpy.ndarray
    :param precision: The presicion required for termination of bisection
    :type precision: float, optional
    :return: The intersection of the expohedron's boundary with the half-line
    :rtype numpy.ndarray
    """
    # 0. Input checks
    n = len(gamma)
    assert n == len(starting_point), "`starting_point` does not have the same length as `gamma`."
    assert n == len(direction), "`direction` does not have the same length as `gamma`."
    assert majorized(starting_point, gamma), "`starting_point` needs to be majorized by `gamma`. Check your inputs or decrease majorization tolerance."

    # direction = direction / np.linalg.norm(direction)  # normalize direction
    # 1. Find upper and lower bound
    k = 1
    while majorized((starting_point + k*direction) / np.sum((starting_point + k*direction)) * np.sum(gamma), gamma):  # We make sure the tested point is in the
        # hyperplane containing the expohedron
        k *= 2
    upper_bound = (starting_point + k*direction) / np.sum((starting_point + k*direction)) * np.sum(gamma)
    lower_bound = starting_point

    # 2. Do bisection
    nb_iterations = 0
    # face = identify_face(gamma, starting_point)
    while True:
        nb_iterations += 1
        center = (upper_bound + lower_bound) / 2
        # center = center / np.sum(center) * np.sum(gamma)
        if majorized(center, gamma, tolerance=precision):  # project center on face's affine subspace
            lower_bound = center
            # face = identify_face(gamma, lower_bound)
            # lower_bound = post_correction(face, lower_bound)
        else:
            upper_bound = center
        if np.all(np.abs(upper_bound - lower_bound) < precision):
            return post_correction(identify_face(gamma, lower_bound), lower_bound)
            # return lower_bound
            # return find_face_intersection_same_ordering(gamma, lower_bound, direction, override_order_constraint=True)
        # elif np.all(np.abs(upper_bound - lower_bound) < precision):
        #     return lower_bound / np.sum(lower_bound) * np.sum(gamma)
        else:
            pass


def identify_face(gamma: np.ndarray, point_on_face: np.ndarray, tolerance: float = LOW_TOLERANCE) -> Face:
    """
        Computes the smallest face of the `gamma`-PBM-expohedron of which `point` is situated

    :param gamma: A vertex of the expohedron
    :type gamma: numpy.array
    :param point_on_face: The point to be examined
    :type point_on_face: numpy.array
    :param tolerance: The allowed tolerance
    :type: float, optional
    :return: The smallest face in which the intersection lies

    A tuple containing:
            (1) The permutation corresponding to an order-preserving zone
            (2) The indices of the splits and the dimensionality of the face
    :rtype: Face
    """
    n = len(gamma)
    assert n == len(point_on_face)

    splits = np.where(np.abs(np.cumsum(np.sort(gamma)) - np.cumsum(np.sort(point_on_face))) < tolerance)
    return Face(gamma, np.argsort(point_on_face), splits[0])


def post_correction(face: Face, point: np.ndarray, tolerance: float = DEFAULT_TOLERANCE) -> np.ndarray:
    """
        Projects a point `point` onto the smallest *affine* subspace that contains the face `face`.
    :param face: The face on whose subspace to project
    :type face: Face
    :param point: The point to project
    :type point: numpy.ndarray
    :param tolerance: The allowed tolerance
    :type tolerance: float, optional
    :return: The projected point, that must now lie on the affine subspace
    :rtype: numpy.ndarray
    """
    vertex_of_face = face.gamma[invert_permutation(face.zone)]
    face_subspace = find_face_subspace(face)
    projected_point = project_on_subspace(point - vertex_of_face, face_subspace.T) + vertex_of_face
    assert face.contains(projected_point), "There has been an error in the projection on a face's subspace"
    return projected_point


def caratheodory_decomposition_pbm_gls(gamma: np.ndarray, point: np.ndarray, tol: float = HIGH_TOLERANCE):
    """
    Finds the Carathéodory decomposition of a point `x` in a PBM-expohedron with vertex `gamma` using the GLS method

    This is done using the GLS procedure
    A non-zero tolerance is necessary to account for numerical imprecision
    :param gamma: The initial vertex of the expohedron
    :type gamma: numpy.ndarray
    :param point: The point to decompose
    :type point: numpy.ndarray
    :param tol: The allowed tolerance
    :type tol: float
    :return: A tuple whose first element is a (matrix whose columns contain the vertices of the decomposition)
        and whose second element is a vector of convex coefficients. We have `point == vertices @ convex_coefficients`.
    :rtype:
    """
    assert majorized(point, gamma), "`x` is not majorized by `gamma`. Only points inside the expohedron can be decomposed as a convex sum of its vertices."

    gamma = np.sort(gamma)  # We work with an increasing permutation of gamma
    n = len(point)
    vertices = np.zeros((n, n))  # Initializing the vertices (empty for now)
    convex_coefficients = np.zeros(n)  # Initializing the convex coefficients (empty for now)

    face = identify_face(gamma, point)

    vertices[:, 0] = gamma[invert_permutation(np.argsort(point))]  # Initialize the initial vertex
    convex_coefficients[0] = 1
    x = point
    dim = identify_face(gamma, x).dim
    for i in np.arange(0, n-1):
        # pdb.set_trace()
        if np.all(np.abs(vertices @ convex_coefficients - point) < tol):
            return convex_coefficients, vertices
        v = vertices[:, i]
        approx_direction = x - v
        direction = project_on_subspace(approx_direction, find_face_subspace(identify_face(gamma, x)).T)
        # direction = approx_direction
        intersection = find_face_intersection(gamma, v, direction)
        intersection = post_correction(identify_face(gamma, intersection), intersection)
        old_ci = convex_coefficients[i]
        convex_coefficients[i] = np.linalg.norm(intersection-x) / np.linalg.norm(intersection-v) * convex_coefficients[i]
        convex_coefficients[i+1] = old_ci - convex_coefficients[i]
        vertices[:, i+1] = gamma[invert_permutation(np.argsort(intersection))]  # Choose a vertex with the same ordering as `u`
        old_x = x
        x = intersection  # `u`, the intersection, is the new point to decompose
        assert identify_face(gamma, x).dim < dim, "At each step, the dimensionality of the face must be reduced"
        dim = identify_face(gamma, x).dim
        if identify_face(gamma, x).dim == 0:
            break  # if we finish on a vertex early, then we break

    # Remove vertices whose coefficients are below a certain threshold
    convex_coefficients[np.where(convex_coefficients < MAX_TOLERANCE)] = 0  # Affect 0 to negligible coefficients
    indices = np.squeeze(np.where(convex_coefficients != 0))
    convex_coefficients = convex_coefficients[indices]
    vertices = vertices[:, indices]

    # Final test
    assert np.abs(np.sum(convex_coefficients) - 1) < tol, "Convex coefficients must sum to 1"
    assert np.all(np.abs(vertices @ convex_coefficients - point) < ULTRA_LOW_TOLERANCE), "Carathéodory decomposition did not work"
    if np.any(np.abs(vertices @ convex_coefficients - point) > DEFAULT_TOLERANCE):
        warnings.warn("Beware, Carathéodory decomposition has reconstruction precision lower than " + str(DEFAULT_TOLERANCE), RuntimeWarning)

    return convex_coefficients, vertices


if __name__ == '__main__':
    gamma = 1 / np.log2(np.arange(0, 3) + 2)
    A = gamma
    F = gamma[[2,1,0]]
    E = gamma[[2,0,1]]
    X = np.ones(3) * np.mean(gamma)
    P = find_face_intersection(gamma, A, X - A)
    Pi = P + 0.5 * (P - X)
    Ci = E + 0.3 * (E - F)
    print(A)
    print(F)
    print(E)
    print(X)
    print(P)
    print(Pi)
    print(Ci)
    print(np.sum(gamma))