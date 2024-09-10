import sparselib
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

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


def L1_vs_time():
	# Initialize solver parameters
	paramsDense = SolverParamsDense()
	paramsSparse = SolverParamsSparse()

	# Standard Galerkin method
	specgalDense = sparselib.SpectralGalerkin(paramsDense)

	# Our sparse method
	specgalSparse = sparselib.SpectralGalerkin(paramsSparse)

	# Evaluate results
	N = 1000
	xs = np.linspace(paramsDense.domain[0], paramsDense.domain[1], N)
	xs = np.expand_dims(xs, axis=1)

	# Compute the propagated uncertainty for our proposed sparse, half-density
	# method, a standard Galerkin approach, and the ground-truth distribution.
	interpDense = np.real(specgalDense.container.grids[0].eval(xs))
	interpSparse = np.power(np.real(specgalSparse.container.grids[0].eval(xs)), 2)

	L1sparse = []
	L1dense = []

	L1sparse.append(np.sum(2 * np.pi * np.abs(interpSparse) / N))
	L1dense.append(np.sum(2 * np.pi * np.abs(interpDense) / N))

	total_time = 1.5
	M = 200
	t = 0
	dt = total_time / M

	print("Computing Lp errors ...")
	pbar = tqdm(total=M)
	for i in range(M):
		t += dt
		specgalSparse.solve(dt)
		specgalDense.solve(dt)

		interpDense = np.real(specgalDense.container.grids[0].eval(xs))
		interpSparse = np.power(np.real(specgalSparse.container.grids[0].eval(xs)), 2)

		L1sparse.append(np.sum(2 * np.pi * np.abs(interpSparse) / N))

		L1dense.append(np.sum(2 * np.pi * np.abs(interpDense) / N))

		pbar.update(1)
	pbar.close()

	L1sparse = np.array(L1sparse)
	L1dense = np.array(L1dense)

	ts = np.linspace(0, total_time, M+1)

	matplotlib.rcParams.update({'font.size': 18})
	fig, ax = plt.subplots(figsize=(18, 6))
	ax.plot(ts, L1sparse, color='#189ab4', linestyle='-', linewidth=3)
	ax.plot(ts, L1dense, color='#fd7f20',  linestyle='--', linewidth=3)
	ax.set_xlabel("Time [s]")
	ax.set_ylabel("$L^1$-norm")
	ax.set_xlim([0, total_time])
	ax.set_ylim([0,1.2])
	plt.gca().legend(('Sparse (Ours)',
					  'Galerkin'))
	fig.tight_layout()
	# plt.grid()
	plt.show()


L1_vs_time()
