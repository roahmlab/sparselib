import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
import os
from matplotlib import cm
from scipy.interpolate import griddata
import config
from config import Domain


# matplotlib.use('Qt5Agg')

class SparseGrid:
    def __init__(self, domain, max_level, dim):
        """
        Sparse grid class using a Fourier basis defined with a hyperbolic
        cross.

        Attributes
        ----------


        Methods
        -------

        """
        self.dim = dim
        self.domain = domain
        self.max_level = max_level

        self.N = []

        self.hyperCross = []            # Hyperbolic Cross
        self.sparseGrid = []            # Corresponding Sparse Grid

        # self.build_sparse_grid()

    def build_sparse_grid(self):
        """
        Builds a hyperbolic cross defined in [1]. First builds the hyperbolic
        cross (2.1) and then builds the corresponding sparse grid (2.3) used
        for computing the weights of the Fourier basis functions.

        [1] MICHAEL DOHLER, STEFAN KUNIS, AND DANIEL POTTS, "NONEQUISPACED HYPERBOLIC 
            CROSS FAST FOURIER TRANSFORM", 2010.
        """

        # Compute translation and scaling relative to [0,1]^d
        translation = self.domain[0, 0]
        scaling = self.domain[0, 1] - self.domain[0, 0]

        # Define constants
        n = 2**(self.max_level-1)

        # Build full tensor grid levels
        nLevel = [np.linspace(0, n, n+1) for _ in range(self.dim)]
        fullCross = np.array(np.meshgrid(*nLevel)).T.reshape(-1, self.dim)

        # Choose subset of grid levels satisfying sparsity condition
        k_mix = np.prod(np.maximum(fullCross + 1, np.ones_like(fullCross)), axis=1)

        # Compute hypercross of Fourier frequencies
        self.hyperCross = self.generate_combinations((fullCross[k_mix <= np.max(n)]))
        self.hyperCross = np.unique(self.hyperCross, axis=0)

        self.N = self.hyperCross.shape[0]

        # Builds the corresponding sparse grid for interpolation over hypercross
        nLevel = [np.linspace(0, k, k+1) for _ in range(self.dim)]
        fullGridLevels = np.array(np.meshgrid(*nLevel)).T.reshape(-1, self.dim)
        level_sums = np.sum(fullGridLevels, axis=1)
        sparseGridLevels = (fullGridLevels[level_sums <= np.max(k)])
        sparseGridLevels = sparseGridLevels[np.argsort(sparseGridLevels.sum(axis=1)),:]

        # Initialize sparse interpolation grid
        pts = np.zeros((1, self.dim))
        grid = [{'level': np.asarray(level)} for level in sparseGridLevels]
        for level in grid:
            h = scaling * 2**(-level['level'])
            idxs = [np.arange(2**(level['level'][d])) for d in range(self.dim)]
            idxs = list(itertools.product(*idxs))

            pt = np.asarray(np.asarray(idxs) * np.asarray(h) + translation)
            pts = np.vstack((pts, pt))

        self.sparseGrid = np.unique(pts[1:, :], axis=0)

    def fit_sparse_grid(self, f):
        """
        Fits sparse grid to given function using the sparse grid interpolation
        scheme.
        
        :param      f:    function to fit
        :type       f:    function handle
        """

        # Function values at all node points
        N = self.sparseGrid.shape[0]    # Number of eval points
        M = self.hyperCross.shape[0]    # Number of basis functions

        A = np.zeros(shape=(N, M), dtype=np.complex128)

        for idx, k in enumerate(self.hyperCross):
            A[:, idx] = self.basis(self.sparseGrid, k)

        self.weights = np.matmul(np.linalg.pinv(A), f(self.sparseGrid))

    def generate_combinations(self, pairs):
        results = []
        for pair in pairs:
            # Create all combinations of negative and positive values for the current pair
            signs = list(itertools.product((1, -1), repeat=len(pair)))
            combinations = [tuple(s * x for s, x in zip(sign, pair)) for sign in signs]
            results.extend(combinations)
        return results

    def eval(self, x):
        """
        Evaluates the sparse grid using the chosen basis at particular points x.
        Uses the chosen basis to build the A matrix (Eq. 16 of [1]) and the stores
        basis weights to compute the interpolation at each point.

        [1] Richard Dennis, "Using a Hyperbolic Cross to Solve Non-linear 
            Macroeconomic Models", 2021.
        
        :param      x:    Interpolation points
        :type       x:    np.array (NxD)
        """
        N = x.shape[0]                  # Number of eval points
        M = self.hyperCross.shape[0]    # Number of basis functions

        A = np.zeros(shape=(N, M), dtype=np.complex128)
        for idx, k in enumerate(self.hyperCross):
            A[:, idx] = self.basis(x, k)

        return np.matmul(A, self.weights)

    def basis(self, x, k):
        """
        Basis function for sparse grid function space.
        
        :param      x:    Evaluation points
        :type       x:    (NxD) np.array
        :param      k:    Evaluation frequency
        :type       k:    float

        :returns:   Basis function evaluation at x
        :rtype:     float
        """
        if not isinstance(k, list) and not isinstance(k, np.ndarray):
            k = [k]
        if not isinstance(x, list) and not isinstance(x, np.ndarray):
            x = np.array([[x]])

        val = 1
        for d in range(self.dim):
            val *= np.exp(1j * k[d] * x[:,d])

        return val

    def basis_der(self, x, k):
        """
        Derivative of basis function for sparse grid function space.
        
        :param      x:    Evaluation points
        :type       x:    (NxD) np.array


        :returns:   Basis function derivative evaluation at x
        :rtype:     float
        """
        if not isinstance(k, list) and not isinstance(k, np.ndarray):
            k = [k]
        if not isinstance(x, list) and not isinstance(x, np.ndarray):
            x = np.array([[x]])

        val = 1
        for d in range(self.dim):
            val *= 1j * k[d] * np.exp(1j * k[d] * x[:,d])
        return val
        
    def marginalize(self, dims, prob=False):
        """
        Compute marginal distribution of dims, integrating out
        additional dimensions from total probability density function.
        
        :param      dims:       The dimensions to compute marginals for
        :type       dims:       list(int)
        """
        print("Marginalizing...")
        if not len(dims) == 2:
            print("Can only marginalize for 2 dimensions at the moment...")
            return

        # Initialize new 1D sparse grid
        spgrid1D = SparseGrid(config.get_inner_product_domain())

        M = 100
        nLevel = [np.linspace(self.domain[dims[0], 0], self.domain[dims[0], 1], M),
                  np.linspace(self.domain[dims[1], 0], self.domain[dims[1], 1], M)]
        coordinates = np.array(np.meshgrid(*nLevel)).T.reshape(-1, 2)

        interp = np.zeros((coordinates.shape[0],))
        for val in spgrid1D.sparseGrid:
            extended_coordinates = np.hstack((coordinates, val * np.ones((coordinates.shape[0],1))))

            if prob:
                interp += np.power(np.real(self.eval(extended_coordinates)), 2)
            else:
                interp += np.real(self.eval(extended_coordinates))

        interp /= spgrid1D.sparseGrid.shape[0]

        xq, yq = np.meshgrid(np.arange(self.domain[dims[0], 0], self.domain[dims[0], 1], 0.01),
                             np.arange(self.domain[dims[1], 0], self.domain[dims[1], 1], 0.01))

        zq = griddata(coordinates[:,dims], interp, (xq, yq))
        fig = plt.figure(figsize =(14, 14))
        ax = plt.axes(projection ='3d')
        surf = ax.plot_surface(xq, yq, zq, cmap=cm.inferno,
                               linewidth=1, antialiased=True)
        plt.show()

        return 
