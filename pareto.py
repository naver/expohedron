"""
    Copyright © 2022 Naver Corporation. All rights reserved.

    This file's contents implement methods used to compute the Pareto set for the fairness-utility trade-off problem
"""

from expohedron import *
from utils import *
from scipy.optimize import minimize, Bounds

UTILITY_PRECISION = 1e-9


class PrecisionError(Exception):

    counter = 0

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
        self.counter += 1


def pareto_curve_pbm(pbm: np.ndarray, relevance_values: np.ndarray, fairness: str, p_norm: float = 2, grouping: np.ndarray = None, verbose: int = 0) -> list:
    """
        Finds the Pareto curve for the multi-objective fairness-utility problem

    :param pbm: The Position-Based Model (PBM) defining the expohedron
    :type pbm: numpy.ndarray
    :param relevance_values: The relevance values of the documents to rank
    :type relevance_values: numpy.ndarray
    :param fairness: The type of fairness to be used. Can be "demographic" for demographic fairness or "meritocratic" for meritocratic fairness
    :type fairness: str
    :param p_norm: The norm to be used for the fairness objective. Only p=2 is currently supported.
    :type p_norm: int, optional
    :param grouping: The group membership matrix of the documents
    :type grouping: numpy.ndarray
    :return: The endpoint of the line segments that constitute the Pareto frontier
    :rtype: list
    """
    # 0. Input checks
    assert p_norm == 2, "Only the 2-norm is currently supported"
    # assert fairness == "demographic", "Only demographic fairness is currently supported"
    assert np.all(pbm >= 0), "Exposures must be positive"
    if grouping is None:
        grouping = np.eye(len(pbm))
    if not np.all(-np.sort(-pbm) == pbm):
        UserWarning("You gave me a PBM that does not have strictly decreasing exposure. Are you sure this is what you want to do ?")

    # 1. Find Fairness starting point and utility endpoint
    vprint("\n-------- Finding Pareto Curve --------\n", verbose)
    vprint("---- Finding Fairness endpoint ----\n", verbose)
    starting_point = fairness_endpoint(pbm, relevance_values, grouping, verbose=verbose)
    vprint("The fairness endpoint is " + str(starting_point), verbose)
    vprint("\n---- Computing the rest of the Pareto curve ----\n", verbose)
    # end_point = -np.argsort(-relevance_values)  # The PRP solution
    # NB: The endpoint is uncertain at this moment, if there are duplicates in the relevance values. However the performance in terms of C-utility of the endpoint is known.
    prp_utility = -np.sort(-relevance_values) @ -np.sort(-pbm)

    # 2. Find segments
    current_point = starting_point
    pareto_endpoints = [current_point]
    while not np.abs(current_point @ relevance_values - prp_utility) < 1e-6:
        vprint("Current utility is " + str(current_point @ relevance_values) + " while the PRP utility is " + str(prp_utility), verbose)
        face = identify_face(pbm, current_point)
        vprint("We are on a face of dimension " + str(face.dim) + " with splits " + str(face.splits), verbose)
        face_orthogonal = find_face_subspace(face).T
        direction = project_on_subspace(relevance_values, face_orthogonal)
        vprint("The projection of the relevance vector on the current face is " + str(direction), verbose)
        current_point = find_face_intersection(pbm, starting_point, direction)
        vprint("Intersection with the expohedron at " + str(current_point), verbose)
        pareto_endpoints.append(current_point)
    return pareto_endpoints


def pareto_curve_pbm_individual(pbm: np.ndarray, relevance_values: np.ndarray, fairness: str, p_norm: float = 2, meritocratic_endpoint: str = "intersection", verbose: int =0) -> list:
    """
        Computes The Pareto curve for individual fairness with a PBM model using the expohedron.

    :param pbm: The Position Based Model (PBM) to be used. Must be ordered from first to last rank, and must be decreasing
    :type pbm: numpy.ndarray
    :param relevance_values: The relevance values of the documents
    :type relevance_values: numpy.ndarray
    :param fairness: The fairness type. "demographic" or "meritocratic" are the two possible options
    :type fairness: str
    :param p_norm: The norm to be used for the fairness objective. Only p=2 is currently implemented
    :type p_norm: float, optional
    :param verbose: How talkative should I be ?
    :type verbose: int, optional
    :return: The Pareto curve in the form of a list of points making up the endpoints of connected line segments
    :rtype: list
    """

    # 0. Input checks
    assert p_norm == 2, "Only p_norm==2 is currently implemented"
    assert len(pbm) == len(relevance_values)
    assert fairness == "demographic" or fairness == "meritocratic"
    assert meritocratic_endpoint == "intersection", "Only the endpoint method is currently implemented"
    assert np.all(-np.sort(-pbm) == pbm), "`pbm` must be decreasing"
    n = len(pbm)

    # 1. Compute Starting point
    vprint("\n-------- Finding Pareto Curve --------\n", verbose)
    vprint("---- Finding Fairness endpoint ----\n", verbose)
    if fairness == "demographic":
        vprint("Demographic fairness chosen. The starting point is the expohedron's barycenter", verbose)
        starting_point = np.ones(n) * np.mean(pbm)  # The starting point is the barycenter
    elif fairness == "meritocratic":
        vprint("Meritocratic fairness chosen", verbose)
        if meritocratic_endpoint == "intersection":
            vprint("Using the intersection method", verbose)
            starting_point = fairness_endpoint_individual_meritocratic_intersection(pbm, relevance_values)
        elif meritocratic_endpoint == "projection":
            raise ValueError("Projection method not yet implemented")

    # 2. Trace the Pareto frontier
    pareto = [starting_point]
    prp_utility = np.sort(pbm) @ np.sort(relevance_values)
    current_point = starting_point
    current_utility = current_point @ relevance_values
    current_face = identify_face(pbm, current_point)
    corrected_point = post_correction(current_face, current_point)
    vprint("The fairness endpoint is " + str(starting_point), verbose)
    vprint("Current utility is " + str(current_point @ relevance_values) + ".\nPRP utility is     " + str(prp_utility), verbose)
    vprint("Current point is " + str(current_point), verbose)
    vprint("On face of dimension " + str(current_face.dim) + " with splits " + str(current_face.splits), verbose)
    vprint("\n---- Computing the rest of the Pareto curve ----\n", verbose)
    k = 0
    while np.abs(current_utility - prp_utility) > UTILITY_PRECISION:
        k += 1
        vprint("\n\n-- S T E P   " + str(k) + " --\n", verbose)
        face_orth = find_face_subspace(current_face)
        direction = project_on_subspace(relevance_values, face_orth.T)
        direction[np.abs(direction) < 1e-13] = 0.
        vprint("The projection of the relevance vector on the current face is " + str(direction), verbose)
        vprint("Current utility is " + str(current_point @ relevance_values) + ".\nPRP utility is     " + str(prp_utility), verbose)
        vprint("Current point is " + str(current_point), verbose)
        vprint("On face of dimension " + str(current_face.dim) + " with splits " + str(current_face.splits), verbose)
        new_point = find_face_intersection(pbm, corrected_point, direction)
        new_face = identify_face(pbm, new_point)
        corrected_point = post_correction(new_face, new_point)
        new_utility = new_point @ relevance_values
        if not new_face.dim < current_face.dim:
            raise PrecisionError("if not face.dim < current_dim","A precision error is likely to have occurred")
        vprint("Intersection with the expohedron at " + str(new_point), verbose)
        pareto.append(new_point)
        current_point = new_point
        current_utility = new_utility
        current_face = new_face
    assert current_utility - prp_utility <= 1e-6, "We get a utility that is higher than PRP, so there is an error somewhere"
    vprint("\n\n--- Utility Endpoint ---\n", verbose)
    vprint("Current utility is " + str(current_point @ relevance_values) + ".\nPRP utility is     " + str(prp_utility), verbose)
    face = identify_face(pbm, current_point)
    vprint("Current point is " + str(current_point), verbose)
    vprint("On face of dimension " + str(face.dim) + " with splits " + str(face.splits) + " in zone " + str(face.zone), verbose)
    return pareto


def fairness_endpoint_individual_meritocratic_intersection(pbm: np.ndarray, relevance_values: np.ndarray) -> np.ndarray:
    """
        Computes the fairness endpoint for meritocratic individual fairness

    :param pbm: The Position Based Model (PBM)
    :type pbm: numpy.ndarray
    :param relevance_values: The vector of relevance values
    :type relevance_values: numpy.ndarray
    :return: The intersection between the true target and the barycenter of the expohedron
    :rtype: numpy.ndarray
    """
    n = len(pbm)
    assert n == len(relevance_values)
    true_target = relevance_values / np.sum(relevance_values) * np.sum(pbm)
    if majorized(true_target, pbm):
        return true_target
    else:
        barycenter = np.ones(n) * np.mean(pbm)
        intersection = find_face_intersection(pbm, barycenter, true_target - barycenter)
        assert identify_face(pbm, intersection).dim < n-1, "The intersection must lie on a face of the expohedron that is not the expohedron itself"
        return intersection


def fairness_endpoint(pbm: np.ndarray, relevance_values: np.ndarray, grouping: np.ndarray = None, fairness: str = "demographic", verbose: int = 0) -> np.ndarray:
    """
        Finds the Pareto-optimal point that minimizes the unfairness for demographic group-fairness

    :param pbm: The Position Based Model (PBM) to be used. It is an array containing the exposures from first to last rank
    :type pbm: numpy.ndarray
    :param relevance_values: The relevance values of the items to be ranked.
    :type relevance_values: numpy.ndarray
    :param grouping: The aggregation matrix to go from item exposure to group exposure
    :type grouping: numpy.ndarray, optional
    :param fairness: The type of fairness to be used
    :type fairness: str, optional
    :return: The maximal-utility point amongst the minimal unfairness points
    :rtype: numpy.ndarray
    """
    n = len(pbm)
    assert n == len(relevance_values), "Not the right amount of relevance values"
    if grouping is None and fairness == "meritocratic":
        return relevance_values / np.sum(relevance_values) * np.sum(pbm)
    assert n == grouping.shape[1], "Grouping matrix has has not as many columns as there are documents"
    assert fairness == "demographic", "For now, only demographic fairness is supported"

    subspace = grouping
    if fairness == "demographic":
        p1 = np.mean(pbm) * np.ones(n)
    elif fairness == "meritocratic":
        p1 = relevance_values / np.sum(relevance_values) * np.sum(pbm)
    else:
        raise ValueError("`fairness` must be either 'demographic' or 'meritocratic'.")
    vprint("Barycenter: " + str(p1), verbose)
    direction = project_on_subspace(relevance_values, subspace)
    vprint("Going into direction " + str(direction) + " after projection on a subspace with " + str(np.linalg.matrix_rank(subspace)) + " orthogonal vectors",
           verbose)
    p2 = find_face_intersection(pbm, p1, direction)
    vprint("Intersecting the expohedron at " + str(p2), verbose)
    majorized(p2, pbm)
    while not np.all(np.abs(p2 - p1) < DEFAULT_TOLERANCE):
        p1 = p2
        face = identify_face(gamma=pbm, point_on_face=p1)
        vprint("Last point is on a face of dimension " + str(face.dim) + " with splits " + str(face.splits), verbose)
        face_orthogonal = find_face_subspace(face)
        subspace = np.concatenate((subspace, face_orthogonal.T), axis=0)
        subspace = orth(subspace.T).T
        vprint("The intersection of the previous subspace with the face's subspace has " + str(np.linalg.matrix_rank(subspace)) + " orthogonal vectors",
               verbose)
        direction = project_on_subspace(relevance_values, subspace)
        vprint("The direction projected on this subspace is " + str(direction), verbose)
        p2 = find_face_intersection(pbm, p1, direction)
        vprint("Intersecting the expohedron at " + str(p2), verbose)
        majorized(p2, pbm)
    vprint("Last two points were identical. Returning last point.", verbose)
    return p2


def pareto_curve_objective_space_individual(pareto_curve_expohedron: list, pbm: np.ndarray, relevance_values: np.ndarray, target: np.ndarray, p_norm: float = 2) -> tuple:
    """
        Takes a Pareto-curve in the expohedron and computes the corresponding Pareto-curve in the objective space

    :param pareto_curve_expohedron:
    :param pbm:
    :param relevance_values:
    :param target:
    :param p_norm:
    :return: The nPBM utility and the unfairness
    """
    n = len(pareto_curve_expohedron)
    utility_vector = np.zeros(n)
    unfairness_vector = np.zeros(n)
    prp_utility = np.sort(pbm) @ np.sort(relevance_values)
    for i in np.arange(0, n):
        v = pareto_curve_expohedron[i]
        utility_vector[i] = v @ relevance_values / prp_utility
        unfairness_vector[i] = compute_unfairness(v, target, p_norm)
    return utility_vector, unfairness_vector


def scalarized_objective_within_pareto_segment(alpha: float, scalarization: float,
                                               point1: np.ndarray, point2: np.ndarray,
                                               target_exposure: np.ndarray,
                                               pbm: np.ndarray,
                                               relevance_vector: np.ndarray,
                                               prp_exposure,
                                               fairness: str) -> float:
    """
        given a convex combination of two exposure vectors, computes the value of the scalarized objective

            min α (-U) + (1-α) F

    :param alpha: The convex combination parameter of the two exposure vectors
    :type alpha: float
    :param scalarization: The scalarization parameter
    :type scalarization: float
    :param point1: An exposure vector
    :type point1: numpy.ndarray
    :param point2: An exposure vector
    :type point2: numpy.ndarray
    :param pbm: The PBM
    :type pbm: numpy.ndarray
    :return: The value of the objective function
    :rtype: float
    """
    exposure = alpha * point1 + (1-alpha) * point2
    return scalarization * (-nutility(exposure, pbm, relevance_vector)) + (1-scalarization) * (np.linalg.norm(exposure - target_exposure) / np.linalg.norm(prp_exposure - target_exposure)) ** 2  # expohedron.nunfairness(exposure, fairness) ** 2


def get_pareto_point_for_scalarization(pareto_curve: list, target_exposure: np.ndarray,
                                       pbm: np.ndarray, prp_exposure, alpha: float,
                                       fairness: str, relevance_vector: np.ndarray) -> tuple:
    """
        Given a Pareto front in an expohedron, and a scalarization parameter `alpha` computes the optimum of the scalarized problem

            min α (-U) + (1-α) F

    :param pareto_curve: The pareto curve in the expohedron
    :type pareto_curve: list
    :param target_exposure: The target exposure vector
    :type target_exposure: numpy.ndarray
    :param expohedron: The expohedron
    :type expohedron: DBNexpohedron
    :param alpha: The scalarization parameter
    :type alpha: float
    :return: The optimal utility, the optimal unfairness and the optimal exposure
    :rtype: Tuple[float, float, numpy.ndarray]
    """

    objective = lambda exposure: alpha * (-nutility(exposure, pbm, relevance_vector)) + \
                                 (1-alpha) * (np.linalg.norm(exposure - target_exposure) / np.linalg.norm(prp_exposure - target_exposure)) ** 2  # expohedron.nunfairness(exposure, fairness) ** 2
    bounds = Bounds(0, 1)
    # Find the line segment on which the optimal exposure lies
    if len(pareto_curve) == 1:  # pathological case
        exposure_opt = pareto_curve[0]
        return nutility(exposure_opt, pbm), (np.linalg.norm(exposure_opt - target_exposure) / np.linalg.norm(prp_exposure - target_exposure)) ** 2, exposure_opt
    for i in np.arange(0, len(pareto_curve)-1):
        o1 = objective(pareto_curve[i])
        o2 = objective(pareto_curve[i+1])
        if o2 > o1: break  # optimal point is in line segment [i, i+1]
    # try:
    #     a = pareto_curve[i]
    # except UnboundLocalError:
    #     a = 1
    sol = minimize(scalarized_objective_within_pareto_segment, 0,
                   (alpha, pareto_curve[i], pareto_curve[i+1], target_exposure, pbm, relevance_vector, prp_exposure, fairness),
                   method="Nelder-Mead", bounds=bounds)
    exposure_opt = sol.x[0] * pareto_curve[i] + (1-sol.x[0]) * pareto_curve[i+1]
    assert objective(exposure_opt) == sol.fun
    return nutility(exposure_opt, pbm, relevance_vector), (np.linalg.norm(exposure_opt - target_exposure) / np.linalg.norm(prp_exposure - target_exposure)) ** 2, exposure_opt


if __name__ == '__main__':
    gamma = np.array([1, 2, 3])
    tol = 1e-9
    rho = np.array([2, 1.5, 2.5])
    computed_pareto_curve = pareto_curve_pbm(gamma, rho, fairness="meritocratic", grouping=None)
    obj = pareto_curve_objective_space_individual(computed_pareto_curve, gamma, rho, rho / np.linalg.norm(rho) * np.linalg.norm(gamma))
    print(obj)
