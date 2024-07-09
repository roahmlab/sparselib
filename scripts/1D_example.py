import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import sparselib

def dynamics(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return -np.sin(2 * x)

def uniform_dense(x):
    """
    Initial Uniform uncertainty, independent of dimension
    
    :param      x:     Coordinates
    :type       x:     np.array
    :param      mu:    Optional mean
    :type       mu:    np.array

    :returns:   initial uncertainty at coords
    :rtype:     np.array
    """
    return np.ones_like(x[:,0]) * 1 / (2 * np.pi)

def uniform_sparse(x):
    """
    Initial Uniform uncertainty, independent of dimension
    
    :param      x:     Coordinates
    :type       x:     np.array
    :param      mu:    Optional mean
    :type       mu:    np.array

    :returns:   initial uncertainty at coords
    :rtype:     np.array
    """
    return np.sqrt(np.ones_like(x[:,0]) * 1 / (2 * np.pi))

def ground_truth(t, x):
	return np.power(np.exp(2*t) * np.power(np.sin(x), 2) + \
                      np.exp(-2*t) * np.power(np.cos(x), 2), -1)

def normalize(ys, N):
	return np.squeeze(ys / (2 * np.pi * np.sum(ys) / N))


class SolverParamsDense():
	max_level: int = 6
	dim: int = 1
	domain: np.ndarray = np.array([0, 2*np.pi])
	funcs: list = [uniform_dense, dynamics]


class SolverParamsSparse():
	max_level: int = 6
	dim: int = 1
	domain: np.ndarray = np.array([0, 2*np.pi])
	funcs: list = [uniform_sparse, dynamics]


# Initialize solver parameters
paramsDense = SolverParamsDense()
paramsSparse = SolverParamsSparse()

# Standard Galerkin method
specgalDense = sparselib.SpectralGalerkin(paramsDense)
specgalDense.solve(1.5)

# Our sparse method
specgalSparse = sparselib.SpectralGalerkin(paramsSparse)
specgalSparse.solve(1.5)

# Evaluate results
N = 1000
xs = np.linspace(paramsDense.domain[0], paramsDense.domain[1], N)
xs = np.expand_dims(xs, axis=1)

# Compute the propagated uncertainty for our proposed sparse, half-density
# method, a standard Galerkin approach, and the ground-truth distribution.
interpDense = normalize(np.real(specgalDense.container.grids[0].eval(xs)), N)
interpSparse = normalize(np.power(np.real(specgalSparse.container.grids[0].eval(xs)), 2), N)
gt = normalize(ground_truth(1.5, xs), N)

matplotlib.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize=(18, 6))
ax.plot(xs, interpSparse, color='#189ab4')
ax.plot(xs, interpDense, color='#fd7f20')
ax.plot(xs, gt, color='green')
ax.set_xlabel("Domain")
ax.set_ylabel("Probability Density")
ax.set_xlim([0, 2*np.pi])
ax.set_ylim(np.minimum(0, np.min(interpDense)), np.maximum(1, np.max(gt)))

fig.tight_layout()
plt.gca().legend(('Half-Densities','Standard Galerkin', 'Ground Truth'))
plt.show()