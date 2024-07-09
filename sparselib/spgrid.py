import itertools
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib import cm
from scipy.interpolate import griddata


class SparseGrid:
    def __init__(self, domain, max_level, dim, logging=False):
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

        self.N = []                     # Number of basis functions
        self.invA = []                  # Matrix for fitting coeffs
        self.hyperCross = []            # Hyperbolic Cross
        self.sparseGrid = []            # Corresponding Sparse Grid

        self.logging = logging

    def build(self):
        """
        Builds a hyperbolic cross defined in [1]. First builds the hyperbolic
        cross (2.1) and then builds the corresponding sparse grid (2.3) used
        for computing the weights of the Fourier basis functions.

        [1] MICHAEL DOHLER, STEFAN KUNIS, AND DANIEL POTTS, "NONEQUISPACED HYPERBOLIC 
            CROSS FAST FOURIER TRANSFORM", 2010.
        """

        self.__buildHyperCross()
        self.__buildSparseGrid(fullGrid=False)
        self.__buildAMatrix()

    def fit(self, f):
        """
        Fits sparse grid to given function using the lienar sparse grid interpolation
        scheme.
        
        :param      f:    function to fit
        :type       f:    function handle
        """
        self.weights = np.matmul(self.invA, f(self.sparseGrid))

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
            A[:, idx] = self.__basis(x, k)

        return np.matmul(A, self.weights)

    def __buildHyperCross(self):
        """
        Builds a hyperbolic cross defined in [1].

        [1] MICHAEL DOHLER, STEFAN KUNIS, AND DANIEL POTTS, "NONEQUISPACED HYPERBOLIC 
            CROSS FAST FOURIER TRANSFORM", 2010.
    
        Parameters:
            d (int): Dimension of the space.
            N (int): Refinement level.
        
        Returns:
            List of tuples representing the indices in the hyperbolic cross.
        """
        def add_index_recursive(dim, max_product, current_product=1, current_index=[]):
            if dim == 0:
                # Base case: If we have filled all dimensions, yield the index
                yield tuple(current_index)
            else:
                for i in range(0, n+1):
                    new_product = current_product * max(i+1, 1)
                    if new_product <= max_product:
                        # Recurse with the next dimension
                        yield from add_index_recursive(dim - 1, max_product, new_product, current_index + [i])

        # Generate all indices for the given dimension and refinement level
        n = 2**(self.max_level-1)
        indices = list(add_index_recursive(self.dim, n))
        
        self.hyperCross = self.__generate_combinations(
                            np.unique(np.asarray(indices), axis=0))
        self.hyperCross = np.unique(np.asarray(self.hyperCross), axis=0)
        self.N = self.hyperCross.shape[0]

    def __buildSparseGrid(self, fullGrid=False):
        """
        Builds the sparse grid (2.3) used for computing the weights of the Fourier 
        basis functions defined in [1]. 

        [1] MICHAEL DOHLER, STEFAN KUNIS, AND DANIEL POTTS, "NONEQUISPACED HYPERBOLIC 
            CROSS FAST FOURIER TRANSFORM", 2010.
        """

        # Compute translation and scaling
        translation = self.domain[0]
        scaling = self.domain[1] - self.domain[0]

        sparseGridLevel = self.max_level + 1

        # Builds the corresponding sparse grid for interpolation over hypercross
        nLevel = [np.linspace(0, sparseGridLevel, sparseGridLevel+1) for _ in range(self.dim)]
        fullGridLevels = np.array(np.meshgrid(*nLevel)).T.reshape(-1, self.dim)
        level_sums = np.sum(fullGridLevels, axis=1)

        if fullGrid:
            sparseGridLevels = fullGridLevels
        else:
            sparseGridLevels = (fullGridLevels[level_sums <= np.max(sparseGridLevel)])

        sparseGridLevels = sparseGridLevels[np.argsort(sparseGridLevels.sum(axis=1)),:]

        # Initialize sparse interpolation grid
        if self.logging:
            print("Building sparse grid...")
            pbar = tqdm(total=len(sparseGridLevels))

        weights = []
        pts = np.zeros((1, self.dim))
        for level in sparseGridLevels:
            h = scaling * 2**(-np.asarray(level))
            idxs = [np.arange(2**(np.asarray(level)[d])) for d in range(self.dim)]
            idxs = list(itertools.product(*idxs))

            # Sparse grid points
            pt = np.asarray(np.asarray(idxs) * np.asarray(h) + translation)
            pts = np.vstack((pts, pt))

            # Corresponding weight for each point for quadrature
            weight = np.prod(2**(-level))
            weights.extend([weight] * len(idxs))

            if self.logging:
                pbar.update(1)

        if self.logging:
            pbar.close()

        self.sparseGrid = np.unique(pts[1:, :], axis=0)
        self.sparseGridWeights = np.array(weights)

    def __buildAMatrix(self):
        """
        Pre-builds the matrix used to fit Fourier coefficients
        """

        N = self.sparseGrid.shape[0]    # Number of eval points
        M = self.hyperCross.shape[0]    # Number of basis functions

        A = np.zeros(shape=(N, M), dtype=np.complex128)
        if self.logging:
            print("Prebuilding matrices...")
            pbar = tqdm(total=M)

        for idx, k in enumerate(self.hyperCross):
            A[:, idx] = self.__basis(self.sparseGrid, k)

            if self.logging:
                pbar.update(1)

        if self.logging:
            pbar.close()

        self.invA = np.linalg.pinv(A)

    def __generate_combinations(self, pairs):
        results = []
        for pair in pairs:
            # Create all combinations of negative and positive values for the current pair
            signs = list(itertools.product((1, -1), repeat=len(pair)))
            combinations = [tuple(s * x for s, x in zip(sign, pair)) for sign in signs]
            results.extend(combinations)
        return results
    
    def __basis(self, x, k):
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