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


omega = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]) + \
            np.random.normal(scale=0.01, size=(6,))
K = np.array([[1.0, 0.1, 0.1, 0.1, 0.1, 0.1],
              [0.1, 1.0, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.1, 1.0, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.1, 1.0, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.1, 1.0, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.1, 1.0]])

def normalize(ys, N):
    return np.squeeze(ys / (2 * np.pi * np.sum(ys) / N))

def monte_carlo_integration(specgal, coords, target_dim=0, N=1000):
    """
    Monte Carlo integration for a 6D function.

    Parameters:
        func (callable): The 6D function to integrate. It should take a numpy array of shape (6,) as input.
        bounds (list of tuples): A list of 6 tuples, each representing the (min, max) bounds for the corresponding dimension.
        N (int): The number of random samples to use for the integration.

    Returns:
        float: The estimated integral of the function over the specified bounds.
    """
    print(f"Marginalizing over dimension {target_dim}...")

    # Generate random samples within the bounds
    random_points = np.random.rand(N, 5)
    
    # Scale points to the specified bounds
    min_bound = 0
    max_bound = 2*np.pi
    scaled_points = random_points * max_bound

    M = coords.shape[0]
    func_values = np.zeros((N, M))

    pbar = tqdm(total=M)
    for i, val in enumerate(coords):
        points = np.insert(scaled_points, target_dim, val, axis=1)
        func_values[:,i] = np.power(np.real(specgal.eval(points)), 2)

        pbar.update(1)

        # Evaluate the function at each sampled point
        # func_values = np.apply_along_axis(specgal.eval, 1, points)
    
    pbar.close()

    # Calculate the volume of the 6D space
    volume = np.power(2*np.pi, 6)
    
    # Estimate the integral
    integral = volume * np.mean(func_values, axis=0)
    
    return normalize(integral, M)

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
                    + K[0,3] * np.sin(x[:,3] - x[:,0]) \
                    + K[0,4] * np.sin(x[:,4] - x[:,0]) \
                    + K[0,5] * np.sin(x[:,5] - x[:,0])

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
                    + K[1,3] * np.sin(x[:,3] - x[:,1]) \
                    + K[1,4] * np.sin(x[:,4] - x[:,1]) \
                    + K[1,5] * np.sin(x[:,5] - x[:,1])

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
                    + K[2,3] * np.sin(x[:,3] - x[:,2]) \
                    + K[2,4] * np.sin(x[:,4] - x[:,2]) \
                    + K[2,5] * np.sin(x[:,5] - x[:,2])

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
                    + K[3,2] * np.sin(x[:,2] - x[:,3]) \
                    + K[3,4] * np.sin(x[:,4] - x[:,3]) \
                    + K[3,5] * np.sin(x[:,5] - x[:,3])

def dynamics5(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return omega[4] + K[4,0] * np.sin(x[:,0] - x[:,4]) \
                    + K[4,1] * np.sin(x[:,1] - x[:,4]) \
                    + K[4,2] * np.sin(x[:,2] - x[:,4]) \
                    + K[4,3] * np.sin(x[:,3] - x[:,4]) \
                    + K[4,5] * np.sin(x[:,5] - x[:,4])

def dynamics6(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return omega[5] + K[5,0] * np.sin(x[:,0] - x[:,5]) \
                    + K[5,1] * np.sin(x[:,1] - x[:,5]) \
                    + K[5,2] * np.sin(x[:,2] - x[:,5]) \
                    + K[5,3] * np.sin(x[:,3] - x[:,5]) \
                    + K[5,4] * np.sin(x[:,4] - x[:,5])

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
    # mu = np.linspace(0, 2*np.pi, 6)
    # mu = np.array([0, 0, 0, 0, 0, 0])
    mu = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])
    # mu = np.linspace(0, 2*np.pi, 6)
    cov = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])

    vals = np.ones((x.shape[0]))
    for d in range(x.shape[1]):
        vals *= 1 / np.sqrt(2 * np.pi * cov[d]) * np.exp(-0.5 * \
                    np.power(x[:,d] - mu[d], 2) / cov[d])

    return np.sqrt(vals)

class SolverParamsSparse():
	max_level: int = 4
	dim: int = 6
	domain: np.ndarray = np.array([0, 2*np.pi])
	funcs: list = [gaussian_uncertainty_sparse,
                   dynamics1, dynamics2, dynamics3,
                   dynamics4, dynamics5, dynamics6]

paramsSparse = SolverParamsSparse()
specgalSparse = sparselib.SpectralGalerkin(paramsSparse, logging=True)

# Propoagation time parameters
total_time = 0.5
M = 10
t = 0
dt = total_time / M
ts = np.linspace(0, total_time, M+1)

# Coordinates for marginalization
x = np.linspace(0, 2*np.pi, 50)

# Results
t1 = monte_carlo_integration(specgalSparse, x, target_dim=0)
t2 = monte_carlo_integration(specgalSparse, x, target_dim=1)
t3 = monte_carlo_integration(specgalSparse, x, target_dim=2)
t4 = monte_carlo_integration(specgalSparse, x, target_dim=3)
t5 = monte_carlo_integration(specgalSparse, x, target_dim=4)
t6 = monte_carlo_integration(specgalSparse, x, target_dim=5)
# plt.plot(t1)
# plt.show()
# plt.plot(t2)
# plt.show()
# plt.plot(t3)
# plt.show()
# plt.plot(t4)
# plt.show()
# plt.plot(t5)
# plt.show()
# plt.plot(t6)
# plt.show()

# coords, t1 = marginalize(specgalSparse, target_dim=0)
# _, t2 = marginalize(specgalSparse, target_dim=1)
# _, t3 = marginalize(specgalSparse, target_dim=2)
# _, t4 = marginalize(specgalSparse, target_dim=3)
# _, t5 = marginalize(specgalSparse, target_dim=4)
# _, t6 = marginalize(specgalSparse, target_dim=5)

xs, ys = np.meshgrid(x, ts)

print("Computing coordinates wrt time ...")
pbar = tqdm(total=M)
for i in range(M):
    t += dt
    specgalSparse.solve(dt)
    
    # Marginalize along each dimension
    # _, t1s = marginalize(specgalSparse, target_dim=0)
    # _, t2s = marginalize(specgalSparse, target_dim=1)
    # _, t3s = marginalize(specgalSparse, target_dim=2)
    # _, t4s = marginalize(specgalSparse, target_dim=3)
    # _, t5s = marginalize(specgalSparse, target_dim=4)
    # _, t6s = marginalize(specgalSparse, target_dim=5)
    
    t1s = monte_carlo_integration(specgalSparse, x, target_dim=0)
    t2s = monte_carlo_integration(specgalSparse, x, target_dim=1)
    t3s = monte_carlo_integration(specgalSparse, x, target_dim=2)
    t4s = monte_carlo_integration(specgalSparse, x, target_dim=3)
    t5s = monte_carlo_integration(specgalSparse, x, target_dim=4)
    t6s = monte_carlo_integration(specgalSparse, x, target_dim=5)

    t1 = np.vstack((t1, t1s))
    t2 = np.vstack((t2, t2s))
    t3 = np.vstack((t3, t3s))
    t4 = np.vstack((t4, t4s))
    t5 = np.vstack((t5, t5s))
    t6 = np.vstack((t6, t6s))
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

fig5 = plt.figure()
ax5 = fig5.add_subplot(111, projection='3d')
ax5.set_xlabel('Frequency')
ax5.set_ylabel('Time [s]')
ax5.set_zlabel('Probability Density')
ax5.set_title('Harmonic Oscillation Frequency vs Time of Fifth Mode')

fig6 = plt.figure()
ax6 = fig6.add_subplot(111, projection='3d')
ax6.set_xlabel('Frequency')
ax6.set_ylabel('Time [s]')
ax6.set_zlabel('Probability Density')
ax6.set_title('Harmonic Oscillation Frequency vs Time of Sixth Mode')

xnew, ynew = np.mgrid[0:2*np.pi:100j, 0:total_time:100j]
tck = interpolate.bisplrep(xs, ys, t1, s=1)
t1 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

tck = interpolate.bisplrep(xs, ys, t2, s=1)
t2 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

tck = interpolate.bisplrep(xs, ys, t3, s=1)
t3 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

tck = interpolate.bisplrep(xs, ys, t4, s=1)
t4 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

tck = interpolate.bisplrep(xs, ys, t5, s=1)
t5 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

tck = interpolate.bisplrep(xs, ys, t6, s=1)
t6 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

ax1.plot_surface(xnew, ynew, t1.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
ax2.plot_surface(xnew, ynew, t2.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
ax3.plot_surface(xnew, ynew, t3.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
ax4.plot_surface(xnew, ynew, t4.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
ax5.plot_surface(xnew, ynew, t5.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
ax6.plot_surface(xnew, ynew, t6.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
plt.show()

np.savez('t1.npz', x=xnew, y=ynew, t1=t1)
np.savez('t2.npz', x=xnew, y=ynew, t1=t2)
np.savez('t3.npz', x=xnew, y=ynew, t1=t3)
np.savez('t4.npz', x=xnew, y=ynew, t1=t4)
np.savez('t5.npz', x=xnew, y=ynew, t1=t5)
np.savez('t6.npz', x=xnew, y=ynew, t1=t6)