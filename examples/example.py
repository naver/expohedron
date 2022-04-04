"""
Copyright © 2022 Naver Corporation. All rights reserved.
"""

"""
This is a kind of example/tutorial made over small controllable data
"""

from pareto import *
import matplotlib.pyplot as plt
import time
from expohedron import *
from billiard import *


# Example data
n_doc = 5
pbm = 1 / np.log(np.arange(0, n_doc) + 2)  # The DCG exposures
np.random.seed(5)
relevance_values = np.random.rand(n_doc)
target = fairness_endpoint_individual_meritocratic_intersection(pbm, relevance_values)

# Compute the Pareto-curve
start = time.time()
pareto_curve = pareto_curve_pbm_individual(pbm, relevance_values, fairness="meritocratic", verbose=1)
end = time.time()
print("Pareto curve was computed in " + str(end - start) + " seconds")
print("\n\nThe Pareto curve is made of the exposure vectors " + str(pareto_curve))
utility_vector, unfairness_vector = pareto_curve_objective_space_individual(pareto_curve, pbm, relevance_values, target)

# Plot
fig, ax = plt.subplots()
ax.plot(utility_vector, unfairness_vector, 'bo-', label='Pareto curve')
# ax.plot(utility_vector_PL, unfairness_vector_PL, 'ro-', label='PL curve')
ax.set(xlabel='nDCG', ylabel='Unfairness',
       title='Objective space')
plt.legend(loc='upper left')
# ax.set_yscale('log')
# ax.set_xlim(min(utility_vector_PL), 1)
# ax.set_ylim(0, max(unfairness_vector_PL))
ax.grid()
# fig.savefig("test.png")
plt.show()


# Decompose the fairness endpoint
tic = time.time()
convex_coefficients, vertices = caratheodory_decomposition_pbm_gls(pbm, pareto_curve[0])
toc = time.time()
print("The Carathéodory decomposition took " + str(toc - tic) + " seconds.")

# Delivery
time_horizon = 1_000  # how many rankings should be delivered
generator = billiard_word(convex_coefficients)
exposure = 0
utility_matrix = np.zeros(time_horizon) * np.nan
unfairness_matrix = np.zeros(time_horizon) * np.nan
idcg = relevance_values @ pbm
for k in np.arange(0, 2*len(pbm)):
  index = next(generator)  # Faire chauffer l'appareil
for t in np.arange(1, time_horizon):
  index = next(generator)
  exposure += vertices[:, index]
  utility_matrix[t] = exposure/t @ relevance_values / idcg
  unfairness_matrix[t] = compute_unfairness(exposure/t, target) / np.sum(pbm)

# Plot
fig, ax = plt.subplots()
ax.plot(np.arange(0,time_horizon), unfairness_matrix, 'bo-', label='Normalized unfairness')

ax.set(ylabel='Normalized unfairness', xlabel='Time',
       title='Unfairness evolution')
plt.legend(loc='upper left')
ax.set_yscale('log')
ax.grid()
plt.show()