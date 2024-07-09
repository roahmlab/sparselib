import numpy as np
import matplotlib
from tqdm import tqdm
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


def Lp_error_vs_time():
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
	interpDense = normalize(np.real(specgalDense.container.grids[0].eval(xs)), N)
	interpSparse = normalize(np.power(np.real(specgalSparse.container.grids[0].eval(xs)), 2), N)
	gt = normalize(ground_truth(0, xs), N)

	L1sparse = []
	L2sparse = []
	Linfsparse = []

	L1dense = []
	L2dense = []
	Linfdense = []

	L1sparse.append(np.sum(2 * np.pi * np.abs(interpSparse - gt) / N))
	L2sparse.append(np.sqrt(np.sum(2 * np.pi * np.power(interpSparse - gt, 2)) / N))
	Linfsparse.append(np.max(np.abs(interpSparse - gt)))

	L1dense.append(np.sum(2 * np.pi * np.abs(interpDense - gt) / N))
	L2dense.append(np.sqrt(np.sum(2 * np.pi * np.power(interpDense - gt, 2)) / N))
	Linfdense.append(np.max(np.abs(interpDense - gt)))

	total_time = 1.5
	M = 100
	t = 0
	dt = total_time / M

	print("Computing Lp errors ...")
	pbar = tqdm(total=M)
	for i in range(M):
		t += dt
		specgalSparse.solve(dt)
		specgalDense.solve(dt)

		interpDense = normalize(np.real(specgalDense.container.grids[0].eval(xs)), N)
		interpSparse = normalize(np.power(np.real(specgalSparse.container.grids[0].eval(xs)), 2), N)
		gt = normalize(ground_truth(t, xs), N)

		L1sparse.append(np.sum(2 * np.pi * np.abs(interpSparse - gt) / N))
		L2sparse.append(np.sqrt(np.sum(2 * np.pi * np.power(interpSparse - gt, 2)) / N))
		Linfsparse.append(np.max(np.abs(interpSparse - gt)))

		L1dense.append(np.sum(2 * np.pi * np.abs(interpDense - gt) / N))
		L2dense.append(np.sqrt(np.sum(2 * np.pi * np.power(interpDense - gt, 2)) / N))
		Linfdense.append(np.max(np.abs(interpDense - gt)))

		pbar.update(1)
	pbar.close()

	L1sparse = np.array(L1sparse)
	L2sparse = np.array(L2sparse)
	Linfsparse = np.array(Linfsparse)

	L1dense = np.array(L1dense)
	L2dense = np.array(L2dense)
	Linfdense = np.array(Linfdense)

	ts = np.linspace(0, total_time, M+1)

	matplotlib.rcParams.update({'font.size': 18})
	fig, ax = plt.subplots(figsize=(18, 6))
	ax.plot(ts, L1sparse, color='#05445e', linestyle='-', marker='^', markevery=10, markersize=10)
	ax.plot(ts, L2sparse, color='#189ab4', linestyle='-', marker='v', markevery=10, markersize=10)
	ax.plot(ts, Linfsparse, color='#75e6da', linestyle='-', marker='*', markevery=10, markersize=10)
	ax.plot(ts, L1dense, color='#fc2e20',  linestyle='--', marker='^', markevery=10, markersize=10)
	ax.plot(ts, L2dense, color='#fd7f20', linestyle='--', marker='v', markevery=10, markersize=10)
	ax.plot(ts, Linfdense, color='#fdb750', linestyle='--', marker='*', markevery=10, markersize=10)
	ax.set_xlabel("Time [s]")
	ax.set_ylabel("$L^p$-norm Error")
	ax.set_xlim([0, total_time])
	plt.gca().legend(('$L^1$-norm (Ours)',
					  '$L^2$-norm (Ours)', 
					  '$L^\\infty$-norm (Ours)', 
					  '$L^1$-norm (Galerkin)', 
					  '$L^2$-norm (Galerkin)', 
					  '$L^\\infty$-norm (Galerkin)'))
	fig.tight_layout()
	plt.show()

def Lp_Error_vs_num_bases():
	N = 1000
	xs = np.linspace(0, 2*np.pi, N)
	xs = np.expand_dims(xs, axis=1)

	L1sparse = []
	L2sparse = []
	Linfsparse = []

	L1dense = []
	L2dense = []
	Linfdense = []

	num_bases_sparse = []
	num_bases_dense = []

	t = 1.0

	pbar = tqdm(total=12)
	for i in range(1, 12):
		# Initialize solver parameters
		paramsDense = SolverParamsDense()
		paramsSparse = SolverParamsSparse()

		paramsSparse.max_level = int(i)
		paramsDense.max_level = int(i)

		# Standard Galerkin method
		specgalDense = sparselib.SpectralGalerkin(paramsDense)
		specgalDense.solve(t)

		# Our sparse method
		specgalSparse = sparselib.SpectralGalerkin(paramsSparse)
		specgalSparse.solve(t)

		interpDense = normalize(np.real(specgalDense.container.grids[0].eval(xs)), N)
		interpSparse = normalize(np.power(np.real(specgalSparse.container.grids[0].eval(xs)), 2), N)
		gt = normalize(ground_truth(t, xs), N)

		L1sparse.append(np.sum(2 * np.pi * np.abs(interpSparse - gt) / N))
		L2sparse.append(np.sqrt(np.sum(2 * np.pi * np.power(interpSparse - gt, 2)) / N))
		Linfsparse.append(np.max(np.abs(interpSparse - gt)))

		L1dense.append(np.sum(2 * np.pi * np.abs(interpDense - gt) / N))
		L2dense.append(np.sqrt(np.sum(2 * np.pi * np.power(interpDense - gt, 2)) / N))
		Linfdense.append(np.max(np.abs(interpDense - gt)))

		num_bases_sparse.append(specgalSparse.container.grids[0].N)
		num_bases_dense.append(specgalDense.container.grids[0].N)

		pbar.update(1)
	pbar.close()

	L1sparse = np.array(L1sparse)
	L2sparse = np.array(L2sparse)
	Linfsparse = np.array(Linfsparse)

	L1dense = np.array(L1dense)
	L2dense = np.array(L2dense)
	Linfdense = np.array(Linfdense)

	num_bases_sparse = np.array(num_bases_sparse)
	num_bases_dense = np.array(num_bases_dense)

	matplotlib.rcParams.update({'font.size': 18})
	fig, ax = plt.subplots(figsize=(18, 6))
	ax.plot(num_bases_sparse, L1sparse, color='#05445e', linestyle='-', marker='^', markersize=10)
	ax.plot(num_bases_sparse, L2sparse, color='#189ab4', linestyle='-', marker='v', markersize=10)
	ax.plot(num_bases_sparse, Linfsparse, color='#75e6da', linestyle='-', marker='*', markersize=10)
	ax.plot(num_bases_dense, L1dense, color='#fc2e20',  linestyle='--', marker='^', markersize=10)
	ax.plot(num_bases_dense, L2dense, color='#fd7f20', linestyle='--', marker='v', markersize=10)
	ax.plot(num_bases_dense, Linfdense, color='#fdb750', linestyle='--', marker='*', markersize=10)
	ax.set_xlabel("Num. of Basis Functions")
	ax.set_ylabel("$L^p$-norm Error")
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlim([1, np.max(num_bases_dense)])
	plt.gca().legend(('$L^1$-norm (Ours)', 
					  '$L^2$-norm (Ours)',
					  '$L^\\infty$-norm (Ours)',
					  '$L^1$-norm (Galerkin)', 
					  '$L^2$-norm (Galerkin)', 
					  '$L^\\infty$-norm (Galerkin)'))
	fig.tight_layout()
	plt.show()


# Lp_error_vs_time()
Lp_Error_vs_num_bases()