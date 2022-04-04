# README

This repository contains code relative to the WSDM'22 paper [Introducing the Expohedron for efficient Pareto-optimal fairness-utility amortizations in repeated rankings](https://dl.acm.org/doi/10.1145/3488560.3498490) by Kletti et al.

## 1. Installation

It is sufficient to clone the repository.
The requirements are listed in the file `requirements.txt`.
If conda is used, they can be installed in a virtual environment `expohedron` as
```shell
conda create --name expohedron
conda activate expohedron
conda install --file requirements.txt
```



## 2. Get started

A complete working example is available in the file `examples/example.py`.

In the following we give examples of some common operations.

### 2.1. Create and expohedron

```python
from expohedron import *

# set parameters
relevance_values = np.array([0.8, 0.8, 0.7, 0.3])
n = len(relevance_values)
dcg = 1 / np.log2(np.arange(2, n+2))

# create expohedron
expohedron = PBMexpohedron(dcg, relevance_values)
print(expohedron)
```

### 2.2. Compute the Pareto curve

```python
from pareto import *

# set parameters
relevance_values = np.array([0.8, 0.8, 0.7, 0.3])
n = len(relevance_values)
dcg = 1 / np.log2(np.arange(2, n+2))

pareto_curve = pareto_curve_pbm_individual(dcg, # position based model (PBM)
                                           relevance_values,  # relevance values
                                           fairness="meritocratic", # set "demographic" for demographic fairness
                                           verbose=1)  # set `verbose=0` for muteness
```

### 2.3. Carathéodory decomposition

```python
from pareto import *

# set parameters
relevance_values = np.array([0.8, 0.8, 0.7, 0.3])
n = len(relevance_values)
dcg = 1 / np.log2(np.arange(2, n+2))

# compute target exposure
target = fairness_endpoint_individual_meritocratic_intersection(dcg, relevance_values)
# Decompose the target exposure as a convex sum of vertices
convex_coefficients, vertices = caratheodory_decomposition_pbm_gls(dcg, target)
```

## 3. License

The license is available in the file [LICENSE](LICENSE)

## 4. References

To cite this repository, please use

```
@inproceedings{kletti_introducing_2022,
	address = {New York, NY, USA},
	series = {{WSDM} '22},
	title = {Introducing the {Expohedron} for {Efficient} {Pareto}-optimal {Fairness}-{Utility} {Amortizations} in {Repeated} {Rankings}},
	isbn = {978-1-4503-9132-0},
	url = {https://doi.org/10.1145/3488560.3498490},
	doi = {10.1145/3488560.3498490},
	urldate = {2022-02-21},
	booktitle = {Proceedings of the {Fifteenth} {ACM} {International} {Conference} on {Web} {Search} and {Data} {Mining}},
	publisher = {Association for Computing Machinery},
	author = {Kletti, Till and Renders, Jean-Michel and Loiseau, Patrick},
	month = feb,
	year = {2022},
	keywords = {ranking, fairness, balanced words, gls, amortization, expohedron, muli-objective optimization, pareto-optimal},
	pages = {498--507},
}

```

## 5. Contact us

Should you have any questions, problems or remarks, do not hesitate to raise an issue or to write to us at [till.kletti@naverlabs.com](mailto:till.kletti@naverlabs.com).
