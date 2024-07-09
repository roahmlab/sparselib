import sparselib
import numpy as np
import matplotlib
import s3dlib.surface as s3d
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import stats
from matplotlib import cm, colorbar
from scipy.integrate import odeint
from scipy.interpolate import griddata
from matplotlib.ticker import LinearLocator


def torusFunc(rtz) :
    r,t,z = rtz
    Z = 0.45 * np.sin(z*np.pi)
    R = r + 0.45 * np.cos(z*np.pi)
    return R,t,Z

def dynamics1(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return 1.0 * np.sin(x[:,2]) + \
            0.2 * np.cos(x[:,1]) + \
            0.5 * np.cos(x[:,0])

def dynamics2(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return 0.5 * np.sin(x[:,2]) + \
            1.0 * np.cos(x[:,1]) + \
            0.5 * np.cos(x[:,1])

def dynamics3(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return 1.0 * np.sin(x[:,2]) + \
            0.5 * np.cos(x[:,1]) + \
            0.5 * np.cos(x[:,2])

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
    mu = np.array([np.pi, np.pi, np.pi])
    cov = np.array([0.2, 0.3, 0.3])

    vals = np.ones((x.shape[0]))
    for d in range(x.shape[1]):
        vals *= 1 / np.sqrt(2 * np.pi * cov[d]) * np.exp(-0.5 * \
                    np.power(x[:,d] - mu[d], 2) / cov[d])

    return np.sqrt(vals)

def gaussian_uncertainty_dense(x):
    """
    Initial Uniform uncertainty, independent of dimension
    
    :param      x:     Coordinates
    :type       x:     np.array
    :param      mu:    Optional mean
    :type       mu:    np.array

    :returns:   initial uncertainty at coords
    :rtype:     np.array
    """
    mu = np.array([np.pi, np.pi, np.pi])
    cov = np.array([0.2, 0.3, 0.3])

    vals = np.ones((x.shape[0]))
    for d in range(x.shape[1]):
        vals *= 1 / np.sqrt(2 * np.pi * cov[d]) * np.exp(-0.5 * \
                    np.power(x[:,d] - mu[d], 2) / cov[d])

    return vals

def plot_torus(zq, filename, title):
    vmin = np.minimum(0, np.min(zq))
    # vmax = np.maximum(1, np.max(zq))
    vmax = np.max(zq)

    cm = plt.get_cmap('magma')
    vals = cm(zq)[:,:,:]
    fig = plt.figure(frameon=False)
    plt.axis('off')
    plt.imshow(zq, cmap=cm, interpolation='bicubic', norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
    # plt.imshow(vals, interpolation='bicubic')
    plt.clim(vmin, vmax)
    plt.savefig('prob.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    torus = s3d.CylindricalSurface(6).map_geom_from_op(torusFunc)
    torus.map_color_from_image('prob.png')

    fig = plt.figure(figsize=plt.figaspect(0.75))
    ax = plt.axes(projection='3d')
    ax.set(xlim=(-1,1), ylim=(-1,1), zlim=(-1,1) )
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_locator(LinearLocator(5))
    minc = torus.bounds['vlim'][0]
    maxc = torus.bounds['vlim'][1]
    ax.add_collection3d(torus)

    # Normalizer 
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax) 

    # creating ScalarMappable 
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm) 
    sm.set_array([])

    plt.colorbar(sm) 
    fig.tight_layout()
    plt.axis('off')
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

def monte_carlo_sample(N):
    t = np.arange(0, 1.51, 0.01)
    mu = np.array([np.pi, np.pi, np.pi])
    cov = np.sqrt(np.array([0.2, 0.3, 0.3]))
    samples = np.random.normal(loc=mu, scale=cov, size=(N,3))
    prop_samples = np.zeros((N, samples.shape[1]))
    # return samples

    pbar = tqdm(total=N)
    for i in range(samples.shape[0]):
        vals = odeint(func, samples[i,:], t)
        prop_samples[i,:] = vals[-1,:]
        pbar.update(1) 
    pbar.close()

    return prop_samples

def func(x, t):
    return [dynamics1_MC(x), dynamics2_MC(x), dynamics3_MC(x)]

def dynamics1_MC(x):
    return 1.0 * np.sin(x[2]) + \
            0.2 * np.cos(x[1]) + \
            0.5 * np.cos(x[0])

def dynamics2_MC(x):
    return 0.5 * np.sin(x[2]) + \
            1.0 * np.cos(x[1]) + \
            0.5 * np.cos(x[1])

def dynamics3_MC(x):
    return 1.0 * np.sin(x[2]) + \
            0.5 * np.cos(x[1]) + \
            0.5 * np.cos(x[2])

def kernel_estimate(samples):
    """
    Plots the gaussian_kde estimate given samples
    """ 
    kde = stats.gaussian_kde(samples.T)

    nLevel = [np.linspace(0, 2*np.pi, 100),np.linspace(0, 2*np.pi, 100)]
    coordinates = np.array(np.meshgrid(*nLevel)).T.reshape(-1, 2)
    z = kde(np.transpose(coordinates))

    dx = 2 * np.pi / 1024
    xq,yq = np.meshgrid(np.arange(0,2*np.pi,dx),
                        np.arange(0,2*np.pi,dx))
    zq = griddata(coordinates, z, (xq, yq), method='cubic')

    M = xq.shape[0]
    zq /= np.sum(zq * (2*np.pi/M)**2)
    return zq

def marginalize(specgal, half=True):
    """
    Compute marginal distribution of dims, integrating out
    additional dimensions from total probability density function.
    
    :param      dims:       The dimensions to compute marginals for
    :type       dims:       list(int)
    """
    print("Marginalizing...")

    # Initialize new 1D sparse grid
    params1D = SolverParams1D()
    spgrid1D = sparselib.SparseGrid(params1D.domain, params1D.max_level, params1D.dim)
    spgrid1D.build()

    M = 20
    nLevel = [np.linspace(params1D.domain[0], params1D.domain[1], M),
              np.linspace(params1D.domain[0], params1D.domain[1], M)]
    coordinates = np.array(np.meshgrid(*nLevel)).T.reshape(-1, 2)

    interp = np.zeros((coordinates.shape[0],))
    for val in spgrid1D.sparseGrid:
        extended_coordinates = np.hstack((coordinates, val * np.ones((coordinates.shape[0],1))))

        if half:
            interp += np.power(np.real(specgal.eval(extended_coordinates)), 2)
        else:
            interp += np.real(specgal.eval(extended_coordinates))

    return coordinates, interp/spgrid1D.sparseGrid.shape[0]

class SolverParams1D():
    max_level: int = 7
    dim: int = 1
    domain: np.ndarray = np.array([0, 2*np.pi])
    funcs: list = [gaussian_uncertainty_sparse, dynamics1]


class SolverParamsSparse():
	max_level: int = 7
	dim: int = 3
	domain: np.ndarray = np.array([0, 2*np.pi])
	funcs: list = [gaussian_uncertainty_sparse, dynamics1, dynamics2, dynamics3]


class SolverParamsDense():
    max_level: int = 5
    dim: int = 3
    domain: np.ndarray = np.array([0, 2*np.pi])
    funcs: list = [gaussian_uncertainty_dense, dynamics1, dynamics2, dynamics3]


global specgalSparse
paramsSparse = SolverParamsSparse()
specgalSparse = sparselib.SpectralGalerkin(paramsSparse, logging=True)
specgalSparse.solve(t=1.5)

# paramsDense = SolverParamsDense()
# specgalDense = sparselib.SpectralGalerkin(paramsDense)
# specgalDense.solve(t=1.5)

coordinates, valsSparse = marginalize(specgalSparse)
# coordinates, valsDense = marginalize(specgalDense)

dx = 2 * np.pi / 1024
xq, yq = np.meshgrid(np.arange(paramsSparse.domain[0], paramsSparse.domain[1], dx),
                     np.arange(paramsSparse.domain[0], paramsSparse.domain[1], dx))

zqSparse = griddata(coordinates, valsSparse, (xq, yq), method='cubic')
# zqDense = griddata(coordinates, valsDense, (xq, yq), method='cubic')

M = xq.shape[0]
zqSparse /= np.sum(zqSparse * (2*np.pi/M)**2)
# zqDense /= (2**2 * np.pi**2 * np.sum(zqDense) / (M * M))

# # Plot spectral methods
plot_torus(zqSparse, 'torusSparseGrid.png', "Sparse Grid Method")
# plot_torus(zqDense, 'torusDenseGrid.png', "Standard Galerkin Method")

# Plot Monte Carllo method
particles = monte_carlo_sample(N=5000)
plot_torus(kernel_estimate(particles[:,:2]), 'torusMonteCarlo.png', "Monte Carlo Method")