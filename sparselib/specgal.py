import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.linalg import pinv, expm

import sparselib


class SpectralGalerkin:
    def __init__(self, params, logging=False):
        """
        Class for spectral Galerkin method.

        ...

        Attributes
        ----------

        Methods
        -------

        """
        self.logging = logging

        self.container = sparselib.GridContainer(params, logging)
        self.container.fit(params)

        self.spgridUncertainty = self.container.grids[0]
        self.spgridVectorField = self.container.grids[1:]

        self.__compute_galerkin_matrix()

    def solve(self, t):
        """
        Propagates the probability density approximation coefficients using the
        spectral Galerkin approach.
        
        :param      t:    Propagation time
        :type       t:    float
        """
        if self.logging:
            print("Propagating...")
            t0 = time.time()

        # Get the weights of the Fourier basis
        coeffs = self.spgridUncertainty.weights

        # Compute Galerkin matrix
        G = -(self.B) / np.power(2 * np.pi, self.spgridUncertainty.dim)

        # Solve ODE
        new_coeffs = expm(G * t) @ np.array(coeffs)

        # Set new coeffs as spgrid weights
        self.spgridUncertainty.weights = new_coeffs

        if self.logging:
            t1 = time.time()
            print('Propagation time: ', t1 - t0)

    def eval(self, x, container_id=0):
        return self.container.grids[container_id].eval(x)

    def __compute_galerkin_matrix(self):
        """
        Calculates the galerkin matrix.
        """
        # Matrices containing sparse grid weights for vector
        # fields.
        D1 = np.zeros((self.spgridUncertainty.N, 
                       self.spgridUncertainty.dim),
                       dtype=np.complex128)
        if self.spgridUncertainty.dim == 1:
            D1 = self.spgridVectorField[0].weights
        else:
            for l in range(self.spgridUncertainty.dim):
                dl1 = self.spgridVectorField[l].weights
                D1[:, l] = dl1

        N = self.spgridUncertainty.N
        hc = np.rint(self.spgridUncertainty.hyperCross).astype('int64')

        tuplook = {}
        for i in range(N):
            tuplook[hc[i, :].tobytes()] = i

        # Compute Galerkin matrix for uncertainty propagation
        B = np.zeros((N, N), dtype=np.complex128)
        cf = 1j * 0.5 * (2 * np.pi)**(self.spgridUncertainty.dim)

        if self.logging:
            pbar = tqdm(total=N*N)
            
        for i in range(N):
            I = hc[i,:]
            for j in range(N):
                J = hc[j,:]
                K_candidate = J - I
                vals_maybe_nz = cf * (I + J)
                k_candidate = tuplook.get(K_candidate.tobytes())
                if k_candidate is not None:
                    B[j, i] = np.dot(vals_maybe_nz, D1[k_candidate,:])

                if self.logging:
                    pbar.update(1)
                    
        if self.logging:
            pbar.close()

        self.B = B
