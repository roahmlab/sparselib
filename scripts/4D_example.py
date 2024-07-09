import sparselib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import stats
from scipy import interpolate
from matplotlib import cm, colorbar
from scipy.integrate import odeint
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from matplotlib.ticker import LinearLocator


omega = [np.pi, np.pi, np.pi, np.pi]
K = np.array([[1.0, 0.1, 0.1, 0.1],
              [0.1, 1.0, 0.1, 0.1],
              [0.1, 0.1, 1.0, 0.1],
              [0.1, 0.1, 0.1, 1.0]])

def normalize(ys, N):
    return np.squeeze(ys / (2 * np.pi * np.sum(ys) / N))

def marginalize(specgal, target_dim=0, num_points=20):
    """
    Compute marginal distribution over the specified target dimension,
    integrating out the additional dimensions from the total probability density function.

    :param      specgal:       The spectral Galerkin object with a method 'eval' that evaluates the PDF.
    :type       specgal:       object
    :param      target_dim:    The dimension to marginalize over (default is the first dimension).
    :type       target_dim:    int
    :param      num_points:    The number of points for interpolation along the target dimension.
    :type       num_points:    int
    :returns:   Tuple containing the coordinates of the marginal distribution and the interpolated values.
    :rtype:     tuple(np.ndarray, np.ndarray)
    """
    print(f"Marginalizing over dimension {target_dim}...")

    class SolverParams1D():
        max_level: int = specgal.container.grids[0].max_level
        dim: int = 1
        domain: np.ndarray = specgal.container.grids[0].domain
        funcs: list = []

    # Determine the number of dimensions in the problem
    total_dim = specgal.spgridUncertainty.dim

    # Create interpolation points along the target dimension
    params1D = SolverParams1D()  # Assuming params1D.domain gives you the range for each dimension
    coordinates = np.linspace(params1D.domain[0], params1D.domain[1], num_points)
    coordinates = np.expand_dims(coordinates, axis=1)  # Shape (num_points, 1)

    # Initialize the result array for the marginal distribution
    interp = np.zeros(coordinates.shape[0])

    # Create the multi-dimensional sparse grid (excluding the target dimension)
    spgrid_params = SolverParamsSparse()  # Assume this gives parameters for the N-1 dimensional grid
    spgridND = sparselib.SparseGrid(spgrid_params.domain, spgrid_params.max_level, total_dim - 1)
    spgridND.build()

    # Loop over all points in the N-1 dimensional sparse grid
    for point, weight in zip(spgridND.sparseGrid, spgridND.sparseGridWeights):
    # for point in spgridND.sparseGrid:
        # Create extended coordinates by inserting the current sparse grid point
        # into every position except the target dimension
        extended_coordinates = np.zeros((coordinates.shape[0], total_dim))
        extended_coordinates[:, target_dim] = coordinates[:, 0]  # Set target dimension
        extended_coordinates[:, np.arange(total_dim) != target_dim] = point  # Set other dimensions

        # Evaluate the PDF at these coordinates and accumulate the results
        interp += weight * np.power(np.real(specgal.eval(extended_coordinates)), 2)

    # Normalize the marginal distribution by the number of points in the N-1 dimensional sparse grid
    interp /= np.sum(spgridND.sparseGridWeights)

    return coordinates, normalize(interp, coordinates.shape[0])

def dynamics1(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return omega[0] + K[0,1] * np.sin(x[:,1] - x[:,0]) \
                    + K[0,2] * np.sin(x[:,2] - x[:,0]) \
                    + K[0,3] * np.sin(x[:,3] - x[:,0])

def dynamics2(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return omega[1] + K[1,0] * np.sin(x[:,0] - x[:,1]) \
                    + K[1,2] * np.sin(x[:,2] - x[:,1]) \
                    + K[1,3] * np.sin(x[:,3] - x[:,1])

def dynamics3(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return omega[2] + K[2,0] * np.sin(x[:,0] - x[:,2]) \
                    + K[2,1] * np.sin(x[:,1] - x[:,2]) \
                    + K[2,3] * np.sin(x[:,3] - x[:,2])

def dynamics4(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return omega[3] + K[3,0] * np.sin(x[:,0] - x[:,3]) \
                    + K[3,1] * np.sin(x[:,1] - x[:,3]) \
                    + K[3,2] * np.sin(x[:,2] - x[:,3])

def gaussian_uncertainty_sparse(x):
    """
    Initial Uniform uncertainty, independent of dimension
    
    :param      x:     Coordinates
    :type       x:     np.array
    :param      mu:    Optional mean
    :type       mu:    np.array

    :returns:   initial uncertainty at coords
    :rtype:     np.array
    """
    mu = np.array([np.pi, np.pi, np.pi, np.pi])
    cov = np.array([0.3, 0.3, 0.3, 0.3])

    vals = np.ones((x.shape[0]))
    for d in range(x.shape[1]):
        vals *= 1 / np.sqrt(2 * np.pi * cov[d]) * np.exp(-0.5 * \
                    np.power(x[:,d] - mu[d], 2) / cov[d])

    return np.sqrt(vals)

class SolverParamsSparse():
	max_level: int = 5
	dim: int = 4
	domain: np.ndarray = np.array([0, 2*np.pi])
	funcs: list = [gaussian_uncertainty_sparse,
                   dynamics1, dynamics2, dynamics3,
                   dynamics4]

paramsSparse = SolverParamsSparse()
specgalSparse = sparselib.SpectralGalerkin(paramsSparse, logging=True)

# Propoagation time parameters
total_time = 0.5
M = 10
t = 0
dt = total_time / M
ts = np.linspace(0, total_time, M+1)

# Results
coords, t1 = marginalize(specgalSparse, target_dim=0)
# plt.plot(t1)
# plt.show()
_, t2 = marginalize(specgalSparse, target_dim=1)
_, t3 = marginalize(specgalSparse, target_dim=2)
_, t4 = marginalize(specgalSparse, target_dim=3)

xs, ys = np.meshgrid(coords, ts)

print("Computing coordinates wrt time ...")
pbar = tqdm(total=M)
for i in range(M):
    t += dt
    specgalSparse.solve(dt)
    
    # Marginalize along each dimension
    _, t1s = marginalize(specgalSparse, target_dim=0)
    _, t2s = marginalize(specgalSparse, target_dim=1)
    _, t3s = marginalize(specgalSparse, target_dim=2)
    _, t4s = marginalize(specgalSparse, target_dim=3)

    t1 = np.vstack((t1, t1s))
    t2 = np.vstack((t2, t2s))
    t3 = np.vstack((t3, t3s))
    t4 = np.vstack((t4, t4s))
    pbar.update(1)
pbar.close()

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_xlabel('Frequency')
ax1.set_ylabel('Time [s]')
ax1.set_zlabel('Probability Density')
ax1.set_title('Harmonic Oscillation Frequency vs Time of First Mode')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Time [s]')
ax2.set_zlabel('Probability Density')
ax2.set_title('Harmonic Oscillation Frequency vs Time of Second Mode')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.set_xlabel('Frequency')
ax3.set_ylabel('Time [s]')
ax3.set_zlabel('Probability Density')
ax3.set_title('Harmonic Oscillation Frequency vs Time of Third Mode')

fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
ax4.set_xlabel('Frequency')
ax4.set_ylabel('Time [s]')
ax4.set_zlabel('Probability Density')
ax4.set_title('Harmonic Oscillation Frequency vs Time of Fourth Mode')

xnew, ynew = np.mgrid[0:2*np.pi:100j, 0:total_time:100j]
tck = interpolate.bisplrep(xs, ys, t1, s=1)
t1 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

tck = interpolate.bisplrep(xs, ys, t2, s=1)
t2 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

tck = interpolate.bisplrep(xs, ys, t3, s=1)
t3 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

tck = interpolate.bisplrep(xs, ys, t4, s=1)
t4 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)


ax1.plot_surface(xnew, ynew, t1.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
ax2.plot_surface(xnew, ynew, t2.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
ax3.plot_surface(xnew, ynew, t3.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
ax4.plot_surface(xnew, ynew, t4.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
plt.show()